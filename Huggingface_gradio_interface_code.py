import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
import json

# ==========================================
# 1. إعدادات النظام والمتغيرات
# ==========================================
REPO_ID = "maherghanem86/Causal-Recommendation_Breaking_the_Filter_Bubble"
NUM_USERS = 69797  
NUM_ITEMS = 10681  
device = torch.device('cpu') 

# ==========================================
# 2. بناء معمارية النموذج
# ==========================================
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=32, mlp_layers=[64, 32, 16], dropout=0.2):
        super(NeuMF, self).__init__()
        self.embedding_user_mf = nn.Embedding(num_users, mf_dim)
        self.embedding_item_mf = nn.Embedding(num_items, mf_dim)
        self.embedding_user_mlp = nn.Embedding(num_users, mlp_layers[0]//2)
        self.embedding_item_mlp = nn.Embedding(num_items, mlp_layers[0]//2)
        
        self.mlp_model = nn.Sequential(
            nn.Linear(mlp_layers[0], mlp_layers[1]),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_layers[1], mlp_layers[2]),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.prediction_layer = nn.Linear(mf_dim + mlp_layers[-1], 1)

    def predict_all(self, u_tensor, all_items_tensor):
        mf_vector = torch.mul(self.embedding_user_mf(u_tensor).expand(len(all_items_tensor), -1), self.embedding_item_mf(all_items_tensor))
        mlp_vector = self.mlp_model(torch.cat([self.embedding_user_mlp(u_tensor).expand(len(all_items_tensor), -1), self.embedding_item_mlp(all_items_tensor)], dim=-1))
        return self.prediction_layer(torch.cat([mf_vector, mlp_vector], dim=-1)).squeeze()

# ==========================================
# 3. تحميل النماذج والقواميس
# ==========================================
base_model = NeuMF(NUM_USERS, NUM_ITEMS).to(device)
causal_model = NeuMF(NUM_USERS, NUM_ITEMS).to(device)

movie_dict = {}
user_bubbles_db = {}

try:
    print("⏳ جاري تحميل النماذج والملفات من المستودع...")
    # تحميل الأوزان
    base_weights = hf_hub_download(repo_id=REPO_ID, filename="neumf_baseline.pth")
    causal_weights = hf_hub_download(repo_id=REPO_ID, filename="neumf_causal.pth")
    base_model.load_state_dict(torch.load(base_weights, map_location=device))
    causal_model.load_state_dict(torch.load(causal_weights, map_location=device))
    base_model.eval()
    causal_model.eval()
    
    # تحميل قواميس الترجمة التي قمتِ برفعها
    movies_path = hf_hub_download(repo_id=REPO_ID, filename="movie_names.json")
    with open(movies_path, "r", encoding="utf-8") as f:
        movie_dict = json.load(f)
        
    bubbles_path = hf_hub_download(repo_id=REPO_ID, filename="user_bubbles.json")
    with open(bubbles_path, "r", encoding="utf-8") as f:
        user_bubbles_db = json.load(f)
        
    print("✅ تم التحميل بنجاح!")
except Exception as e:
    print(f"❌ حدث خطأ أثناء التحميل: {e}")

# ==========================================
# 4. دالة التنبؤ للواجهة
# ==========================================
def generate_recommendations(user_id):
    try:
        u_idx = int(user_id)
        if u_idx < 0 or u_idx >= NUM_USERS:
            return "رقم غير صالح.", "رقم غير صالح.", "رقم غير صالح."
            
        all_items_tensor = torch.arange(NUM_ITEMS).to(device)
        u_tensor = torch.tensor([u_idx]).to(device)
        
        # استخراج الفقاعة
        user_bubble = user_bubbles_db.get(str(u_idx), "غير متوفرة في العينة. يعتمد النظام على التفضيلات العامة.")
        formatted_bubble = f"🔍 الفقاعة المعلوماتية المكتشفة: {user_bubble}"

        # توقعات النظامين
        with torch.no_grad():
            scores_base = base_model.predict_all(u_tensor, all_items_tensor).numpy()
            top_base = np.argsort(scores_base)[::-1][:10]
            
            scores_causal = causal_model.predict_all(u_tensor, all_items_tensor).numpy()
            top_causal = np.argsort(scores_causal)[::-1][:10]
            
        # ترجمة الأرقام إلى أسماء أفلام
        base_output = "\n\n".join([f"🎬 {movie_dict.get(str(item), f'فيلم غير معروف (ID: {item})')}" for item in top_base])
        causal_output = "\n\n".join([f"✨ {movie_dict.get(str(item), f'فيلم غير معروف (ID: {item})')}" for item in top_causal])
        
        return formatted_bubble, base_output, causal_output
        
    except ValueError:
        return "الرجاء إدخال رقم صحيح.", "خطأ", "خطأ"

# ==========================================
# 5. تصميم واجهة Gradio
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🎓 تطبيق الاستدلال السببي لمعالجة التحيز في أنظمة التوصية")
    gr.Markdown("هذه الواجهة تقارن بين نظام التوصية التقليدي (الذي يعزز فقاعة التصفية) والنظام السببي المقترح (الذي يعزز استقلالية المستخدم).")
    
    with gr.Row():
        user_input = gr.Textbox(label="أدخل رقم المستخدم (مثال: 0, 42, 60, 105)", placeholder="اكتب رقم المستخدم هنا...")
    btn = gr.Button("توليد التوصيات المقارنة 🚀", variant="primary")
    out_bubble = gr.Textbox(label="التشخيص المسبق لحالة المستخدم", lines=2)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ❌ النظام التقليدي (NeuMF Baseline)")
            gr.Markdown("*يميل إلى حصر المستخدم في تفضيلاته السابقة (فقاعة التصفية).*")
            out_base = gr.Textbox(label="التوصيات (العالم المرصود)", lines=15)
            
        with gr.Column():
            gr.Markdown("### ✅ النظام السببي (Causal Autonomy NeuMF)")
            gr.Markdown("*يطبق التدخل السببي لكسر التنميط وتعزيز التنوع والاستقلالية.*")
            out_causal = gr.Textbox(label="التوصيات (العالم المضاد للواقع)", lines=15)

    btn.click(fn=generate_recommendations, inputs=user_input, outputs=[out_bubble, out_base, out_causal])

demo.launch(theme=gr.themes.Soft(), ssr_mode=False)