import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import json
import requests
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=API_KEY)
gemini = GenerativeModel("gemini-2.0-flash-exp")

# url = os.getenv("SUPABASE_URL")
# key = os.getenv("SUPABASE_KEY")
# supabase: Client = create_client(url, key)

# def fetch_data_from_supabase():
#     signed_url = os.getenv("STORAGE_URL")

#     response = requests.get(signed_url)

#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise Exception(f"Failed to fetch data. Status Code: {response.status_code}")

# data = fetch_data_from_supabase()

data = [
  {
    "input": "What are the placement statistics for PES University?",
    "output": "PES University boasts strong placement records, particularly in the Computer Science and Engineering (CSE) department. Approximately 88% of students from this course were placed, with the highest package reaching 63 LPA, the lowest at 8 LPA, and an average package of 12 LPA. Top recruiters include Intel, Texas Instruments, Microsoft, Schneider, Schindler, and Google."
  },
  {
    "input": "How does the Electronic City (EC) campus compare to the Ring Road (RR) campus at PES University?",
    "output": "Both campuses offer similar academic programs and share placements, ensuring equal opportunities for students. The RR campus features newer infrastructure, a larger number of clubs, and hosts bigger fests. In contrast, the EC campus provides a more spacious environment and is known for its strong technical clubs. Ultimately, the choice between campuses may come down to personal preferences regarding infrastructure and extracurricular activities."
  },
  {
    "input": "What are the pros and cons of studying at PES University?",
    "output": "Pros include excellent placement opportunities, especially for CSE students, a modern and regularly updated curriculum, a competitive and inspiring peer group, and a strategic location in Bengaluru, offering proximity to numerous tech companies. Cons encompass high tuition fees, a rigorous academic environment that can be stressful for some, and disparities in attention and resources between different branches, with CSE often receiving the most focus."
  },
  {
    "input": "Is there a significant difference in placement opportunities between the RR and EC campuses?",
    "output": "Placements are pooled for both campuses, meaning students from both RR and EC have equal access to the same placement opportunities. Companies do not differentiate between campuses during recruitment, ensuring a level playing field for all students."
  },
  {
    "input": "How does the faculty at PES University support student learning?",
    "output": "The faculty at PES University is known for being helpful and interactive, with a strong emphasis on clearing doubts and ensuring students understand the material. They teach contemporary subjects such as Data Science, Data Structures and Algorithms (DSA), and Java, preparing students for current industry demands."
  },
  {
    "input": "What is the campus life like at PES University?",
    "output": "Campus life at PES University is vibrant, with excellent library facilities, well-maintained classrooms accommodating around 70 students each, and good hostel facilities. The university also offers various extracurricular activities, including technical clubs and cultural events, contributing to a holistic student experience."
  },
  {
    "input": "Are there scholarship opportunities available at PES University?",
    "output": "Yes, PES University offers a scholarship program where students can receive up to 40% of their fees back if they are among the top 20% of their batch. This initiative encourages academic excellence and helps alleviate the financial burden on high-performing students."
  },
  {
    "input": "How does the curriculum at PES University stay relevant to industry trends?",
    "output": "PES University maintains a modern curriculum that is regularly updated to include current subjects like Big Data, Machine Learning, and Application Development, alongside core subjects. This approach ensures that students are well-prepared for the evolving demands of the industry."
  },
  {
    "input": "What are the hostel facilities like at PES University?",
    "output": "The hostel facilities at PES University are considered good, offering a decent quality of food, including non-vegetarian options like chicken thrice a week. The hostels provide a comfortable living environment conducive to student life."
  },
  {
    "input": "How does PES University support research activities?",
    "output": "PES University houses the Crucible Of Research and Innovation (CORI), a multi-disciplinary research center inaugurated by Dr. C. N. R. Rao. It offers research facilities and dedicated staff who carry out research in various areas, promoting a culture of innovation among students and faculty."
  },
  {
    "input": "What transportation options are available for students at PES University?",
    "output": "PES University provides transportation facilities with multiple routes covering various parts of Bengaluru. The service includes designated boarding points, specific timings, and associated charges, ensuring convenient and safe travel for students to and from the campus."
  },
  {
    "input": "How does PES University ensure a competitive peer environment?",
    "output": "PES University attracts a talented and competitive student body, fostering an environment that encourages academic and extracurricular excellence. Students are inspired by their peers' achievements, motivating them to strive for success."
  },
  {
    "input": "What are the library facilities like at PES University?",
    "output": "The university boasts excellent library facilities, providing students with access to a vast collection of resources to support their academic pursuits. The well-maintained libraries offer a conducive environment for study and research."
  },
  {
    "input": "How does PES University integrate practical learning into its curriculum?",
    "output": "PES University places a strong emphasis on projects, with each subject often having an associated project component. This approach ensures that students gain hands-on experience, preparing them for real-world applications and enhancing their problem-solving skills."
  },
  {
    "input": "What is the fee structure at PES University?",
    "output": "The fee structure at PES University is considered high by some, especially for those admitted through management quotas. However, the university offers scholarship programs that can offset a portion of the fees for high-performing students."
  },
  {
    "input": "What are the extracurricular opportunities available at PES University?",
    "output": "PES University offers a variety of clubs and student organizations, including technical, cultural, and sports clubs. Frequent hackathons, coding competitions, and student-driven initiatives provide ample opportunities for extracurricular engagement."
  },
  {
    "input": "How strict is the attendance policy at PES University?",
    "output": "PES University enforces a strict attendance policy requiring students to maintain at least 75% attendance, with minimal exceptions. Falling below this threshold may result in academic penalties."
  },
  {
    "input": "Are students from all departments equally placed?",
    "output": "While the CSE department enjoys the best placement opportunities, students from branches like ECE and Mechanical also receive good offers. However, non-CS branches generally have fewer high-paying job opportunities."
  },
  {
    "input": "Does PES University offer interdisciplinary opportunities?",
    "output": "Yes, students at PES University can participate in interdisciplinary projects, research collaborations, and elective courses that allow them to explore multiple domains beyond their primary field of study."
  },
  {
    "input": "What is the student-teacher ratio at PES University?",
    "output": "The student-teacher ratio varies by department, but it is generally around 50:1 in most branches, ensuring a relatively balanced academic interaction between faculty and students."
  }
]

model = SentenceTransformer("all-MiniLM-L6-v2")

qa_pairs = [(entry["input"], entry["output"]) for entry in data]

embeddings = np.array([model.encode(q) for q, _ in qa_pairs], dtype="float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_answer(question, top_k=3):
    query_embedding = np.array([model.encode(question)], dtype="float32")
    distances, indices = index.search(query_embedding, top_k)
    
    matches = [(qa_pairs[idx][0], qa_pairs[idx][1], distances[0][i]) 
              for i, idx in enumerate(indices[0])]

    relevant_matches = []
    combined_answers = []
    
    for q, a, d in matches:
        if d < 1.2: 
            weight = 1.0
            relevant_matches.append((q, a, d, "High"))
            combined_answers.append(a)
        elif d < 1.8: 
            weight = 0.7
            relevant_matches.append((q, a, d, "Moderate"))
            combined_answers.append(a)
    
    if combined_answers:
        combined_answer = " ".join(combined_answers)
    else:
        combined_answer = ""

    return combined_answer

def is_pesu_related(question):
    pesu_keywords = [
        'pesu', 'pes', 'university', 'campus', 'ec', 'rr', 'electronic city', 
        'ring road', 'placement', 'faculty', 'department', 'hostel', 'fee', 
        'scholarship', 'admission', 'pessat', 'bangalore', 'bengaluru'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in pesu_keywords)

class RAGResponse(BaseModel):
    answer: str
    source_context: str
    confidence_score: float

def ragg(question):
    retrieved_answer = retrieve_answer(question)
    query_embedding = np.array([model.encode(question)], dtype="float32")
    distances, indices = index.search(query_embedding, 1)
    confidence_score = 1.0 / (1.0 + distances[0][0])

    if retrieved_answer:
        prompt = f"""You are a knowledgeable assistant with access to a database of accurate information. 
        Answer the question using the provided context as your primary source, and supplement with your knowledge when needed.

        Context from database: {retrieved_answer}
        Question: {question}

        Instructions:
        1. Primary Source Usage:
           - First, use the information from the provided context
           - Highlight specific details from the context when relevant
           - If multiple pieces of information are provided, combine them coherently

        2. Knowledge Integration:
           - Supplement the context with additional relevant information
           - Ensure all information is accurate and up-to-date
           - If there's any conflict between context and current knowledge, acknowledge it

        3. Response Format:
           - Be direct and concise
           - Use plain text without special formatting
           - Structure the response logically
           - If uncertain about any details, acknowledge the limitations

        Remember: The context provided is from a reliable source, so prioritize this information in your response.
        """
    else:
        prompt = f"""You are a knowledgeable assistant. While no specific context was found in our database, 
        please provide a comprehensive answer based on your understanding.

        Question: {question}

        Instructions:
        - Provide accurate, up-to-date information
        - Be direct and concise
        - Use plain text without special formatting
        - If the question is about PESU or related topics, mention that specific information might be limited
        - Acknowledge any uncertainties
        """

    response = gemini.generate_content(prompt)
    result = RAGResponse(
        answer=response.text,
        source_context=retrieved_answer if retrieved_answer else "No relevant context found - Using general knowledge",
        confidence_score=float(confidence_score)
    ).dict()
    return result