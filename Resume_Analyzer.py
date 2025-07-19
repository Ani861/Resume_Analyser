import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
import fitz

import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------- Core Functions ----------------
def extract_text_from_pdf(file):
    text = ""
    pdf_file = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf_file:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def named_entities(text):
    chunks = ne_chunk(pos_tag(word_tokenize(text)))
    entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entities.append(" ".join(c[0] for c in chunk))
    return entities

def extract_projects_and_experience(text):
    """Extract projects and experience sections from resume"""
    text_lower = text.lower()
    
    # Enhanced patterns for project and experience sections
    project_patterns = [
        r'projects?\s*:?\s*(.*?)(?=experience|education|skills|certification|$)',
        r'academic\s+projects?\s*:?\s*(.*?)(?=experience|education|skills|certification|$)',
        r'personal\s+projects?\s*:?\s*(.*?)(?=experience|education|skills|certification|$)',
        r'key\s+projects?\s*:?\s*(.*?)(?=experience|education|skills|certification|$)'
    ]
    
    experience_patterns = [
        r'experience\s*:?\s*(.*?)(?=education|projects?|skills|certification|$)',
        r'work\s+experience\s*:?\s*(.*?)(?=education|projects?|skills|certification|$)',
        r'professional\s+experience\s*:?\s*(.*?)(?=education|projects?|skills|certification|$)',
        r'employment\s+history\s*:?\s*(.*?)(?=education|projects?|skills|certification|$)'
    ]
    
    projects_text = ""
    experience_text = ""
    
    # Extract projects
    for pattern in project_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            projects_text += match.group(1) + " "
    
    # Extract experience
    for pattern in experience_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            experience_text += match.group(1) + " "
    
    return projects_text.strip(), experience_text.strip()

def enhanced_skill_extraction(text, skill_set, projects_text="", experience_text=""):
    """Enhanced skill extraction considering projects and experience with weighted scoring"""
    
    # Convert to lowercase for matching
    full_text = (text + " " + projects_text + " " + experience_text).lower()
    words = set(word_tokenize(full_text))
    skill_set_lower = [skill.lower() for skill in skill_set]
    
    matched_skills = []
    skill_contexts = {}
    
    # Technology clusters for better matching
    tech_clusters = {
        'react': ['reactjs', 'react.js', 'react js'],
        'node': ['nodejs', 'node.js', 'node js'],
        'javascript': ['js', 'javascript', 'ecmascript'],
        'python': ['python', 'py'],
        'java': ['java', 'jdk', 'jre'],
        'sql': ['mysql', 'postgresql', 'sqlite', 'sql server'],
        'html': ['html5', 'html'],
        'css': ['css3', 'css', 'stylesheet'],
        'mongodb': ['mongo', 'mongodb'],
        'docker': ['containerization', 'docker'],
        'git': ['github', 'gitlab', 'version control'],
        'aws': ['amazon web services', 'ec2', 's3', 'lambda'],
        'azure': ['microsoft azure', 'azure cloud'],
        'machine learning': ['ml', 'artificial intelligence', 'ai'],
        'tensorflow': ['tf', 'tensorflow'],
        'deep learning': ['neural networks', 'cnn', 'rnn']
    }
    
    for skill in skill_set_lower:
        skill_found = False
        context_info = []
        
        # Direct skill match
        if skill in words:
            matched_skills.append(skill)
            skill_found = True
        
        # Check technology clusters
        if skill in tech_clusters:
            for variant in tech_clusters[skill]:
                if variant in full_text:
                    if not skill_found:
                        matched_skills.append(skill)
                        skill_found = True
        
        # Context analysis
        if skill_found:
            if skill in projects_text.lower():
                context_info.append("Project")
            if skill in experience_text.lower():
                context_info.append("Experience")
            if skill in text.lower() and skill not in projects_text.lower() and skill not in experience_text.lower():
                context_info.append("Resume")
            
            skill_contexts[skill] = context_info
    
    return list(set(matched_skills)), skill_contexts

def calculate_enhanced_match_score(resume_text, job_skills, projects_text="", experience_text=""):
    """Calculate match score with project and experience weighting"""
    if not job_skills:
        return 0.0, {}
    
    matched_skills, contexts = enhanced_skill_extraction(resume_text, job_skills, projects_text, experience_text)
    
    # Weighted scoring
    score = 0
    max_score = len(job_skills) * 3  # Max 3 points per skill
    
    for skill in matched_skills:
        skill_score = 1  # Base score
        if skill in contexts:
            if "Experience" in contexts[skill]:
                skill_score += 2  # +2 for work experience
            elif "Project" in contexts[skill]:
                skill_score += 1  # +1 for projects
        score += skill_score
    
    final_score = (score / max_score) * 100
    return round(min(final_score, 100.0), 2), contexts

# ---------------- Simplified UI ----------------
st.set_page_config(page_title="CareerLens", page_icon="📌", layout="wide")

# Enhanced Custom CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #4a5568;
        font-size: 1.4rem;
        font-weight: 300;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out;
        opacity: 0.8;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        transform: translateY(0);
        transition: all 0.3s ease;
        animation: slideInUp 0.8s ease-out;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .metric-container:hover {
        transform: translateY(-10px);
        box-shadow: 0 30px 60px rgba(102, 126, 234, 0.4);
    }
    
    .metric-container h3 {
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-container h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .skill-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        transform: translateY(0);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .skill-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
    }
    
    .missing-skill-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(238, 90, 82, 0.3);
        transform: translateY(0);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .missing-skill-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(238, 90, 82, 0.4);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
        animation: fadeInUp 0.8s ease-out;
        min-height: 120px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #2d3748;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    .upload-area {
        background: rgba(255, 255, 255, 0.9);
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .upload-area:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        border: 2px dashed #667eea;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 2rem 0;
        border-radius: 1px;
        opacity: 0.3;
    }
    
    /* Fix for empty containers */
    .element-container {
        min-height: auto !important;
    }
    
    /* Ensure proper spacing */
    .row-widget {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🚀 CareerLens</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>✨ Your Smart AI-Powered Resume Analyzer with Advanced Analytics</p>", unsafe_allow_html=True)

# Main container to prevent empty spaces
with st.container():
    col1, col2 = st.columns([1, 1])
    
    with col1:
       
        st.markdown('<h3 class="section-header">🔍 Select Job Role</h3>', unsafe_allow_html=True)
        category = st.selectbox("Choose a domain", ["IT", "Accounting/Finance/Management"], 
                               help="Select your career domain to get tailored analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
       
        st.markdown('<h3 class="section-header">📤 Upload Resume</h3>', unsafe_allow_html=True)
        resume_file = st.file_uploader("📎 Drop your Resume here (PDF only)", type=["pdf"],
                                      help="Upload your resume in PDF format for AI analysis")
        st.markdown('</div>', unsafe_allow_html=True)

# Job roles dictionary
job_roles = {
    "IT": {
        "Software Engineer": ["Python", "Java", "C++", "Git", "SQL", "OOP", "REST", "APIs", "Agile", "Scrum", "Linux"],
        "Machine Learning Engineer": ["Python", "Machine Learning", "Numpy", "Pandas", "Scikit-learn", "TensorFlow", "Deep Learning", "NLP", "Data Preprocessing"],
        "Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "Redux", "Responsive Design", "UI/UX", "Bootstrap"],
        "Backend Developer": ["Node.js", "Express.js", "MongoDB", "SQL", "REST APIs", "Authentication", "Docker", "Cloud", "CI/CD"],
        "Data Analyst": ["SQL", "Excel", "Python", "Data Visualization", "Tableau", "Power BI", "Statistics", "Business Analysis"],
        "Full Stack Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express.js", "MongoDB", "SQL", "REST APIs", "Git", "Docker", "CI/CD"],
        "Data Engineer/Cloud Engineer": ["Python", "SQL", "ETL", "Data Pipelines", "Apache Spark", "Hadoop", "Airflow", "AWS", "Big Data", "Data Warehousing","Azure"],
        "iOS Developer": ["Swift", "Objective-C", "Xcode", "iOS SDK", "Core Data", "UIKit", "Auto Layout", "REST APIs", "Git", "App Store Deployment"],
        "Android Developer": ["Kotlin", "Java", "Android Studio", "Jetpack Compose", "XML Layouts", "Room DB", "Firebase", "Material Design", "REST APIs", "Git"]
    },
    "Accounting/Finance/Management": {
        "Accountant": ["Accounting", "Bookkeeping", "Tally", "ERP", "Taxation", "Invoicing", "MS Excel", "GST", "Reconciliation"],
        "Financial Analyst": ["Finance", "Investment", "Risk Management", "Financial Analysis", "Portfolio Management", "Budgeting", "Forecasting", "Excel", "Valuation"],
        "Auditor": ["Auditing", "Compliance", "Internal Controls", "Risk Assessment", "Documentation", "Tax Laws", "Accounting Standards"],
        "Operations Manager": ["Operations Management", "Strategic Planning", "Business Development", "Team Management", "KPI", "Process Improvement"]
    }
}

job_options = list(job_roles[category].keys())
selected_job = st.selectbox("Select Job Role", job_options, 
                           help="Choose the specific role you're targeting")
job_skills = job_roles[category][selected_job]

if resume_file is not None:
    with st.spinner("🧠 AI is analyzing your resume with advanced algorithms..."):
        raw_text = extract_text_from_pdf(resume_file)
        cleaned_resume = clean_text(raw_text)
        projects_text, experience_text = extract_projects_and_experience(raw_text)

    st.success("✅ Resume analysis completed successfully!")

    # Move Named Entities expander to the top
    with st.expander("🧠 Named Entities & Key Information Extracted", expanded=False):
        entities = named_entities(raw_text)
        if entities:
            st.markdown("**🏷️ Identified Entities:**")
            entities_html = ""
            for entity in entities[:10]:  # Show first 10 entities
                entities_html += f"<span class='skill-badge' style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #2d3748;'>{entity}</span>"
            st.markdown(entities_html, unsafe_allow_html=True)
        else:
            st.info("🤖 No specific named entities were detected in your resume.")

    # Enhanced metrics with glassmorphism cards - Remove empty spaces
    st.markdown("---")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
         
            st.metric("🎯 Selected Role", selected_job, help="Your target position")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
          
            st.metric("🛠️ Expected Skills", f"{len(job_skills)}", help="Total skills required for this role")
            st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced skill matching
    matched_skills, skill_contexts = enhanced_skill_extraction(raw_text, job_skills, projects_text, experience_text)
    missing_skills = [skill for skill in job_skills if skill.lower() not in [s.lower() for s in matched_skills]]
    
    enhanced_score, contexts = calculate_enhanced_match_score(cleaned_resume, job_skills, projects_text, experience_text)

    # Beautiful match score display
    st.markdown("---")
    
    # Remove extra columns, just center the metric
    with st.container():
        st.markdown(f"""
        <div class='metric-container pulse' style='margin-left:auto; margin-right:auto; max-width:400px;'>
            <h3>🎯 AI Match Score</h3>
            <h1>{enhanced_score}%</h1>
            <p>Advanced skills matching analysis</p>
            <div style="margin-top: 1rem;">
                <small>Based on AI-powered content analysis</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
           
            st.metric("✅ Skills Matched", f"{len(matched_skills)}", 
                     delta=f"{len(matched_skills)}/{len(job_skills)}", 
                     help="Number of required skills found in your resume")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            
            completion_rate = round((len(matched_skills) / len(job_skills)) * 100) if job_skills else 0
            st.metric("📊 Completion Rate", f"{completion_rate}%", 
                     help="Percentage of required skills you possess")
            st.markdown('</div>', unsafe_allow_html=True)

    # Beautiful skills analysis
    st.markdown("---")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            
            st.markdown('<h3 class="section-header">✅ Your Matched Skills</h3>', unsafe_allow_html=True)
            if matched_skills:
                skills_html = ""
                for skill in matched_skills:
                    skills_html += f"<span class='skill-badge'>{skill.title()}</span>"
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.markdown("🤔 No matching skills found. Consider updating your resume!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
           
            st.markdown('<h3 class="section-header">❌ Skills to Develop</h3>', unsafe_allow_html=True)
            if missing_skills:
                missing_html = ""
                for skill in missing_skills:
                    missing_html += f"<span class='missing-skill-badge'>{skill}</span>"
                st.markdown(missing_html, unsafe_allow_html=True)
                st.markdown("<small>💡 Focus on learning these skills to improve your match score!</small>", unsafe_allow_html=True)
            else:
                st.markdown("🎉 Congratulations! You have all required skills!")
            st.markdown('</div>', unsafe_allow_html=True)

    # Fixed Pie Chart with proper matplotlib configuration
    st.markdown("---")
    
    with st.container():
       
        st.markdown('<h2 style="text-align: center; color: #2d3748; margin-bottom: 2rem;">📈 Skills Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        if matched_skills or missing_skills:
            # Create a more beautiful pie chart with fixed RGBA values
            labels = ['Matched Skills', 'Skills to Develop']
            sizes = [len(matched_skills), len(missing_skills)]
            colors = ['#4facfe', '#ff6b6b']
            explode = (0.05, 0.05)  # explode slices
            
            # Set matplotlib to use a compatible backend and fix RGBA issue
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(2,2))  # Reduced size
            
            # Use solid white background instead of transparent
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                             startangle=45, colors=colors, explode=explode,
                                             textprops={'fontsize': 14, 'weight': 'bold', 'color': '#2d3748'},
                                             shadow=True)
            
            # Beautify the pie chart
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')
                
            ax.set_title('Skills Match Analysis', fontsize=18, fontweight='bold', 
                        color='#2d3748', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Clear the figure to prevent memory issues
            plt.clf()
            plt.close()
            
            # Add insights below chart
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem; padding: 1rem; 
                       background: rgba(79, 172, 254, 0.1); border-radius: 10px;">
                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">📊 Quick Insights</h4>
                <p style="color: #4a5568; margin: 0;">
                    You have <strong>{len(matched_skills)} out of {len(job_skills)}</strong> required skills. 
                    {f"Focus on developing <strong>{len(missing_skills)}</strong> more skills to become a perfect match!" if missing_skills else "You're a perfect match! 🎉"}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)