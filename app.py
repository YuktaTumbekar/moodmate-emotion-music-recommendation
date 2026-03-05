# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ===================== UI =====================
# st.markdown("""
# <style>
# body {
#     background: linear-gradient(135deg, #1e1e2f, #2b1055);
# }
# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(135deg, #1e1e2f, #2b1055);
# }
# </style>
# """, unsafe_allow_html=True)


# # st.title("🎵 MoodMate")
# # st.write("Emotion-based music recommendation system")

# st.markdown("""
# <h1 style='text-align: center; color: #f5f5f5; font-size: 48px;'>
# 🎵 MoodMate
# </h1>
# <p style='text-align: center; color: #cfcfcf; font-size: 18px;'>
# Emotion-based music recommendation system
# </p>
# <hr>
# """, unsafe_allow_html=True)


# img = st.camera_input("Capture your face")
# analyze = st.button("🎧 Analyze Mood & Recommend Songs")

# # ===================== LOAD MODEL =====================
# @st.cache_resource
# def load_cnn_model():
#     return load_model("cnn_model.h5")

# model = load_cnn_model()

# emotion_labels = [
#     'angry', 'disgust', 'fear',
#     'happy', 'sad', 'surprise', 'neutral'
# ]

# # ===================== EMOTION DETECTION =====================
# def detect_emotion_from_streamlit_image(img):
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )

#     # Convert Streamlit image → OpenCV image
#     img_array = np.frombuffer(img.getbuffer(), np.uint8)
#     frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     if frame is None:
#         return None

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0:
#         return None

#     x, y, w, h = faces[0]
#     face = gray[y:y+h, x:x+w]

#     face = cv2.resize(face, (48, 48))
#     face = face / 255.0
#     face = face.reshape(1, 48, 48, 1)

#     pred = model.predict(face)
#     return emotion_labels[np.argmax(pred)]

# # ===================== LOAD SONG DATA =====================
# @st.cache_data
# def load_song_data():
#     df = pd.read_csv("cleaned_music_dataset.csv")
#     df["tags"] = df["tags"].astype(str)
#     return df

# songs_df = load_song_data()

# tfidf = TfidfVectorizer(stop_words="english")
# song_vectors = tfidf.fit_transform(songs_df["tags"])

# emotion_music_map = {
#     "angry": "metal hard_rock heavy_metal nu_metal punk punk_rock grunge thrash_metal",
#     "disgust": "gothic gothic_metal experimental",
#     "fear": "ambient experimental electronic trip_hop",
#     "happy": "pop dance britpop indie_pop upbeat energetic",
#     "neutral": "alternative indie classic_rock soundtrack new_wave post_punk",
#     "sad": "acoustic mellow chillout piano beautiful love singer_songwriter emo",
#     "surprise": "electronic experimental dance funk psychedelic progressive_rock"
# }

# def recommend_songs(emotion, top_n=10):
#     query = emotion_music_map.get(emotion, "")
#     query_vec = tfidf.transform([query])

#     scores = cosine_similarity(query_vec, song_vectors).flatten()
#     top_indices = scores.argsort()[::-1][:top_n]

#     return songs_df.iloc[top_indices][["track_id", "name", "artist"]]

# # ===================== PIPELINE =====================
# if img is not None and analyze:
#     with st.spinner("Analyzing emotion..."):
#         emotion = detect_emotion_from_streamlit_image(img)

#     if emotion is None:
#         st.error("❌ No face detected")
#     else:
#         # st.success(f" Detected Expression: {emotion}")
#         st.markdown(f"""
#         <div style="
#         background: rgba(255,255,255,0.08);
#         padding: 20px;
#         border-radius: 15px;
#         text-align: center;
#         margin-top: 20px;
#         box-shadow: 0 0 20px rgba(0,0,0,0.3);
#         ">
#         <h2 style="color:#ffd369;">😄 Detected Emotion</h2>
#         <h1 style="color:#ffffff; letter-spacing:2px;">
#         {emotion.upper()}
#         </h1>
#         </div>
#         """, unsafe_allow_html=True)


#         recommendations = recommend_songs(emotion)
#         # st.subheader("🎧 Top 10 Recommended Songs")
#         # st.dataframe(recommendations)
#         st.markdown("""
#         <h2 style="color:#00ffd5; margin-top:30px;">
#         🎧 Top 10 Recommended Songs
#         </h2>
#         """, unsafe_allow_html=True)

#         st.dataframe(
#             recommendations,
#             use_container_width=True
#         )

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components


st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ===================== GLOBAL AESTHETIC THEME =====================
# st.markdown("""
# <style>
# [data-testid="stAppViewContainer"] {
#     background: #0f1117;
#     color: #ffffff;
# }

# [data-testid="stHeader"] {
#     background: rgba(0,0,0,0);
# }

# button {
#     border-radius: 12px !important;
#     padding: 0.6em 1.2em !important;
#     font-size: 14px !important;
# }

# hr {
#     border: none;
#     height: 1px;
#     background: rgba(255,255,255,0.1);
#     margin: 30px 0;
# }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f4efe9, #ede6dc);
    color: #3e342a;
}

/* Remove Streamlit header bar */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Buttons */
button {
    border-radius: 16px !important;
    padding: 0.6em 1.6em !important;
    font-size: 14px !important;
    background-color: #e7dccf !important;
    color: #3e342a !important;
    border: 1px solid #d6c8b8 !important;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #d8cfc3;
    margin: 32px 0;
}
</style>
""", unsafe_allow_html=True)



# ===================== TITLE =====================

st.markdown("""
<h1 style="
    text-align:center;
    font-weight:400;
    letter-spacing:4px;
    color:#3e342a;
    margin-bottom:6px;
">
MOODMATE
</h1>
<p style="
    text-align:center;
    color:#7a6a58;
    font-size:25px;
    margin-top:0;
">
Emotion-based music recommendation
</p>
<hr>
""", unsafe_allow_html=True)


# # ===================== CAMERA & BUTTON =====================
img = st.camera_input("Capture your face")
analyze = st.button("Analyze mood")




# ===================== LOAD MODEL =====================
@st.cache_resource
def load_cnn_model():
    return load_model("cnn_model.h5")

model = load_cnn_model()

emotion_labels = [
    'angry', 'disgust', 'fear',
    'happy', 'sad', 'surprise', 'neutral'
]

# ===================== EMOTION DETECTION =====================
def detect_emotion_from_streamlit_image(img):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    img_array = np.frombuffer(img.getbuffer(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)

    pred = model.predict(face)
    return emotion_labels[np.argmax(pred)]

# ===================== LOAD SONG DATA =====================
@st.cache_data
def load_song_data():
    df = pd.read_csv("cleaned_music_dataset.csv")
    df["tags"] = df["tags"].astype(str)
    return df

songs_df = load_song_data()

tfidf = TfidfVectorizer(stop_words="english")
song_vectors = tfidf.fit_transform(songs_df["tags"])

emotion_music_map = {
    "angry": "metal hard_rock heavy_metal nu_metal punk punk_rock grunge thrash_metal",
    "disgust": "gothic gothic_metal experimental",
    "fear": "ambient experimental electronic trip_hop",
    "happy": "pop dance britpop indie_pop upbeat energetic",
    "neutral": "alternative indie classic_rock soundtrack new_wave post_punk",
    "sad": "acoustic mellow chillout piano beautiful love singer_songwriter emo",
    "surprise": "electronic experimental dance funk psychedelic progressive_rock"
}

def recommend_songs(emotion, top_n=10):
    query = emotion_music_map.get(emotion, "")
    query_vec = tfidf.transform([query])

    scores = cosine_similarity(query_vec, song_vectors).flatten()
    top_indices = scores.argsort()[::-1][:top_n]

    return songs_df.iloc[top_indices][["track_id", "name", "artist"]]

def render_song_table(df):
    rows = ""
    for _, row in df.iterrows():
        rows += f"""
        <tr>
            <td>{row['track_id']}</td>
            <td><strong>{row['name']}</strong></td>
            <td>{row['artist']}</td>
        </tr>
        """

def render_song_table(df):
    rows = ""
    for _, row in df.iterrows():
        rows += f"""
        <tr>
            <td>{row['track_id']}</td>
            <td><strong>{row['name']}</strong></td>
            <td>{row['artist']}</td>
        </tr>
        """

    html = f"""
    <div style="
        background:#fffaf4;
        border-radius:18px;
        padding:24px;
        border:1px solid #e0d6c9;
        box-shadow:0 6px 18px rgba(160,140,120,0.25);
        font-family: system-ui, -apple-system, BlinkMacSystemFont;
    ">
        <table style="
            width:100%;
            border-collapse:collapse;
            font-size:18px;
            color:#3e342a;
        ">
            <thead>
                <tr style="border-bottom:1px solid #e0d6c9;">
                    <th style="text-align:left; padding:10px;">Track ID</th>
                    <th style="text-align:left; padding:10px;">Song</th>
                    <th style="text-align:left; padding:10px;">Artist</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """

    components.html(html, height=420)




# ===================== PIPELINE =====================
if img is not None and analyze:
    with st.spinner("Analyzing facial expression…"):
        emotion = detect_emotion_from_streamlit_image(img)

    if emotion is None:
        st.error("No face detected")
    else:
        # ===================== EMOTION CARD =====================

        # st.markdown(f"""
        # <div style="
        #     background: rgba(255,255,255,0.06);
        #     backdrop-filter: blur(12px);
        #     padding: 28px;
        #     border-radius: 18px;
        #     text-align: center;
        #     margin-top: 30px;
        #     border: 1px solid rgba(255,255,255,0.08);
        # ">
        # <p style="
        #     color:#9fa4b5;
        #     font-size:12px;
        #     letter-spacing:1.5px;
        #     margin-bottom:6px;
        # ">
        # DETECTED EMOTION
        # </p>
        # <h1 style="
        #     color:#ffffff;
        #     font-weight:500;
        #     letter-spacing:3px;
        #     margin:0;
        # ">
        # {emotion.upper()}
        # </h1>
        # </div>
        # """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            background: rgba(255,252,248,0.95);
            padding: 32px;
            border-radius: 22px;
            text-align: center;
            margin-top: 32px;
            border: 1px solid #e0d6c9;
            box-shadow: 0 6px 20px rgba(160,140,120,0.25);
        ">
        <p style="
            color:#8b7a66;
            font-size:12px;
            letter-spacing:1.5px;
            margin-bottom:6px;
        ">
        DETECTED EMOTION
        </p>
        <h1 style="
            color:#3e342a;
            font-weight:500;
            letter-spacing:3px;
            margin:0;
        ">
        {emotion.upper()}
        </h1>
        </div>
        """, unsafe_allow_html=True)


        # ===================== SONGS =====================
        recommendations = recommend_songs(emotion)

        # st.markdown("""
        # <p style="
        #     color:#9fa4b5;
        #     font-size:12px;
        #     letter-spacing:1.5px;
        #     margin-top:40px;
        # ">
        # RECOMMENDED TRACKS
        # </p>
        # """, unsafe_allow_html=True)

        st.markdown("""
        <p style="
            color:#7a6a58;
            font-size:12px;
            letter-spacing:1.5px;
            margin-top:44px;
        ">
        RECOMMENDED TRACKS
        </p>
        """, unsafe_allow_html=True)


        # st.dataframe(
        #     recommendations,
        #     use_container_width=True
        # )

        render_song_table(recommendations)
