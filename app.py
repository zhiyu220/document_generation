from flask import Flask, request, jsonify, render_template
import os
import openai
import google.generativeai as genai
import fitz  # PyMuPDF
import chromadb
from langchain_openai import OpenAIEmbeddings
import re
import random
from dotenv import load_dotenv

# ==== Flask 設定 ====
app = Flask(__name__)

# ==== 載入 .env ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# ==== 向量資料庫 (備用學系特色檢索) ====
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="department_info")

# ==== 連接 PostgreSQL 資料庫 ====
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ==== 代碼對應表 ====
CODE_MAPPING = {
    "A": "修課紀錄", "B": "書面報告", "C": "實作作品", "D": "自然科學探究",
    "E": "社會領域探究", "F": "高中自主學習計畫", "G": "社團活動經驗", "H": "幹部經驗",
    "I": "服務學習經驗", "J": "競賽表現", "K": "非修課成果作品", "L": "檢定證照",
    "M": "特殊優良表現", "N": "多元表現綜整心得", "O": "高中學習歷程反思",
    "P": "學習動機", "Q": "未來學習計畫"
}

# ==== 分項表與Prompt ====
SECTION_MAPPING = {
    "ABC": {  
        "codes": ["A", "B", "C", "D", "E"],
        "prompt": "請撰寫一篇課程學習成果，強調學習歷程與探索精神，並與學系關聯性相結合。"
    },
    "N": {  
        "codes": ["F", "G", "H", "I", "J", "K", "L", "M"],
        "prompt": "請了解學系專業與選材理念後，撰寫強調自主學習的多元表現綜整心得，包含 100 字引言、3個段落(其中每個段落需有一個標題，總結該段落的內容)，以及總結，並利用使用者輸入的具體事例舉例每件事的能力成長與省思。內容與學系密切相關。"
    },
    "O": {  
        "codes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
        "prompt": "請撰寫高中學習歷程反思，強調學習經驗、挑戰與成長，並舉出失敗與反省以展現學習態度。"
    },
    "P": {  
        "codes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
        "prompt": "請撰寫學習動機，強調與學系高度相關的興趣與企圖心，並適當佐證。",
        "needs_department_features": True
    },
    "Q": {  
        "codes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
        "prompt": "請撰寫未來計畫，分成三個階段：(1)入學前想學習的課程與技能 (2)就讀期間計畫修習的課程 (3)畢業後的職業發展。",
        "needs_department_features": True
    }
}

# ==== 風格Prompt ====
STYLE_PROMPTS = {
    "formal": "請以正式且學術性的語氣撰寫，邏輯嚴謹，避免過度修飾。",
    "casual": "請以口語化的語氣撰寫，內容稍微白話]，但仍保持專業度。",
    "concise": "請以簡潔精煉的語言撰寫，避免冗長，聚焦於核心要點。",
    "detailed": "請撰寫深入分析的內容，提供具體論據和例子，表達完整。",
}

# ==== 檢索學系特色 (從向量資料庫) ====
def retrieve_department_features():
    results = collection.query(query_texts=["學系特色", "學系培養目標"], n_results=3)
    return "\n".join(results["documents"][0]) if "documents" in results and results["documents"] else "未找到學系特色。"

# ==== 獲取所有學校與學系 ====
@app.route("/get_schools_departments", methods=["GET"])
def get_schools_departments():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT university, department FROM application_guidelines")
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    schools_departments = {}
    for university, department in results:
        if university not in schools_departments:
            schools_departments[university] = []
        schools_departments[university].append(department)

    return jsonify(schools_departments)

# ==== 學系備審指引查詢 ====
@app.route("/get_guidelines", methods=["POST"])
def get_guidelines():
    data = request.json
    university = data.get("university")
    department = data.get("department")

    if not university or not department:
        return jsonify({"error": "請選擇學校與學系"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT section_code, section_name, requirements, content, department_features 
        FROM application_guidelines 
        WHERE university=%s AND department=%s
    """, (university, department))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return jsonify({"error": "該學校學系尚無備審資料"}), 400

    sections = {}
    for row in results:
        sections[row[0]] = {
            "name": row[1],
            "requirements": row[2],
            "content": row[3]
        }
        if row[4]:  # 讀取學系特色
            department_features = row[4]

    return jsonify({
        "university": university,
        "department": department,
        "sections": sections,
        "department_features": department_features or "未提供學系特色"
    })

# ==== 進入 index ====
@app.route("/")
def index():
    return render_template("index.html")

# ==== 生成段落 ====
@app.route("/generate_paragraph", methods=["POST"])
def generate_paragraph():
    data = request.get_json()
    
    section_code = data.get("section_code")
    user_inputs = data.get("user_inputs", {})
    department_name = data.get("department_name", "未知學系")
    #風格
    style = data.get("style", "formal")  # 預設風格為正式
    #字數
    min_words = data.get("min_words", 800)
    max_words = data.get("max_words", 1000)
    word_count = random.randint(min_words, max_words)
    #調整程度
    adjust_percentage = 30
    
    if section_code not in SECTION_MAPPING:
        return jsonify({"error": "無效的段落代碼"}), 400
    
     # **獲取學系特色**
    department_features = get_department_features(university, department)
    if not department_features:
        department_features = retrieve_department_features()  # 從向量資料庫補充

    # **獲取撰寫指引 (section_prompt)**
    section_prompt = SECTION_MAPPING[section_code]["prompt"]
    
    input_text = "\n".join(f"{CODE_MAPPING.get(code, code)}: {text}" for code, text in user_inputs.items() if text)

    # **使用GPT初次生成**
    first_prompt = f"""
    你是一名高中生，正在申請{department_name}，請根據以下內容撰寫 {section_prompt}。

    **請務必遵守以下規則，嚴格按照使用者提供的內容進行撰寫，不得編造資訊**：
    1.**只能使用以下學系特色，不得新增額外內容**：{department_features}
    2.**只能使用使用者提供的內容，不得自行發揮**：{input_text}
    3.**字數範圍：約 {word_count} 字**
    4.**風格要求：{STYLE_PROMPTS.get(style, STYLE_PROMPTS['formal'])}**
    5.**不得杜撰經歷、不得添加未提供的競賽、活動、學術研究等內容**。
    6.**內容應清晰、邏輯合理、段落流暢，並忠實呈現使用者輸入的重點**。

    請根據這些要求，產生一段符合申請需求的內容。
    """

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": first_prompt}],
        temperature=0.6
    )
    first_output = response.choices[0].message.content.strip()
    final_output = first_output

    # **進行 N 次優化**
    N = 1
    for _ in range(N):
        refine_prompt = f"""
        這是先前生成的內容：
        {final_output}

        請根據以下要求進一步優化：
        1. **請調整語句，使表達方式稍有不同，但仍然保持相同核心內容**
        2. **請確保字數範圍在 {word_count} 字左右**
        3. **請保持 {STYLE_PROMPTS.get(style, STYLE_PROMPTS['formal'])} 的語氣**
        4. **請讓語句更流暢，避免過於生硬**
        5.**只能使用以下學系特色，不得新增額外內容**：{department_features}
        6.**只能使用使用者提供的內容，不得自行發揮**：{input_text}
        7.**變更幅度：約 {adjust_percentage}%**
        """
         # 動態調整 temperature
        temperature_value = 0.3 + (adjust_percentage / 100) * 0.4
        if adjust_percentage > 30:
            refine_prompt += "\n請嘗試替換一些詞語，使表達方式更加生動。"
        if adjust_percentage > 50:
            refine_prompt += "\n請嘗試變換句子結構，使內容更加流暢自然。"
        if adjust_percentage > 70:
            refine_prompt += "\n請嘗試用不同的方式表達相同意思，使表達方式多樣化。"
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=temperature_value
        )
        
        generated_text = response.choices[0].message.content.strip()
        final_output = generated_text
        
    # **使用 Gemini 進行語氣優化**
    gemini_prompt = f"""
    這是一篇大學申請備審資料，請幫助我優化語氣，使其更加自然、流暢，並符合申請文件的語氣要求。

    目前的內容：
    {final_output}

    請確保：
    1. **保持原始內容的邏輯與重點，不可新增資訊**
    2. **保持使用者選擇的風格：{STYLE_PROMPTS.get(style, STYLE_PROMPTS['formal'])}**
    3. **修正冗長、不自然或不流暢的部分**
    4. **讓表達方式更有說服力**

    優化後的內容：
    """

    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    gemini_response = gemini_model.generate_content(
        gemini_prompt, 
        generation_config={"temperature": 0.7}
    )
    improved_output = gemini_response.text.strip()

    return jsonify({
        "generation_steps": N,
        "style": style,
        "adjust_percentage": adjust_percentage,
        "generated_text": improved_output
    })

if __name__ == "__main__":
    app.run(debug=True)
