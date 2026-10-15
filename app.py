import streamlit as st
import yt_dlp
import librosa
import soundfile as sf
import numpy as np
import os
import shutil
import re
import subprocess
import imageio_ffmpeg

st.set_page_config(page_title="Beat Workflow Hub", page_icon="🎛️", layout="wide")

# Custom VST / Dark Hardware Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #121417;
        color: #E2E8F0;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1A1D24 !important;
        color: #FFFFFF !important;
        border: 1px solid #2D3748 !important;
        border-radius: 6px;
    }
    .stButton button {
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        opacity: 0.9;
    }
    div[data-testid="stMetricValue"] {
        color: #00D2FF !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🎛️ Beat Workflow Hub — VST Edition")
st.markdown("Zero-distraction studio command center with robust MP3 extraction, tempo/key detection, loop trimming, and AI stem separation.")

st.sidebar.header("Studio Preferences")
download_mode = st.sidebar.selectbox("Output Format:", ["Audio Only (MP3)", "Direct Media Download"])
auto_rename = st.sidebar.checkbox("Smart Auto-Rename ([BPM]bpm_[Key])", value=True)

tab_download, tab_stems, tab_trim, tab_history = st.tabs(["📥 Download & Analyze", "🧬 AI Stem Splitter", "✂️ Loop Trimmer", "📜 Session Crate Log"])

with tab_download:
    url = st.text_input("YouTube Beat URL:", placeholder="Paste link here...")
    
    if url:
        if st.button("🚀 Process & Analyze Beat", type="primary"):
            output_dir = os.path.expanduser("~/Downloads")
            downloaded_file = None
            raw_title = "Beat"
            
            with st.spinner("Pulling and converting audio to MP3..."):
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                
                # Bypasses 403 Forbidden by routing through ios/tv embedded clients
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'ffmpeg_location': ffmpeg_path,
                    'extractor_args': {'youtube': {'player_client': ['android', 'ios']}},
                    'nocheckcertificate': True,
                    'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                }

                if download_mode == "Audio Only (MP3)":
                    ydl_opts['postprocessors'] = [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '320',
                    }]

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        raw_title = info.get('title', 'beat')
                        filename = ydl.prepare_filename(info)
                        
                        if download_mode == "Audio Only (MP3)":
                            downloaded_file = os.path.splitext(filename)[0] + ".mp3"
                        else:
                            downloaded_file = filename
                except Exception as ex:
                    st.error(f"Download Error: {ex}")

            if downloaded_file and os.path.exists(downloaded_file):
                with st.spinner("Analyzing Tempo & Key..."):
                    try:
                        y, sr = librosa.load(downloaded_file, duration=60)
                        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                        detected_bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
                        
                        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                        chroma_mean = np.mean(chroma, axis=1)
                        major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
                        minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
                        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                        
                        best_score = -1
                        detected_key = "Unknown"
                        for i in range(12):
                            maj_corr = np.corrcoef(chroma_mean, np.roll(major_template, i))[0, 1]
                            if maj_corr > best_score:
                                best_score = maj_corr
                                detected_key = f"{pitches[i]} Major"
                            min_corr = np.corrcoef(chroma_mean, np.roll(minor_template, i))[0, 1]
                            if min_corr > best_score:
                                best_score = min_corr
                                detected_key = f"{pitches[i]} Minor"
                    except Exception:
                        detected_bpm = 0.0
                        detected_key = "N/A"
                
                final_filename = downloaded_file
                if auto_rename and detected_bpm > 0:
                    clean_title = re.sub(r'[\\/*?:"<>|]', "", raw_title).strip()
                    ext = os.path.splitext(downloaded_file)[1]
                    formatted_name = f"{int(round(detected_bpm))}bpm_{detected_key.replace(' ', '')}_{clean_title}{ext}"
                    final_filename = os.path.join(output_dir, formatted_name)
                    if os.path.exists(downloaded_file) and downloaded_file != final_filename:
                        os.rename(downloaded_file, final_filename)

                st.session_state.history.insert(0, {
                    "Title": raw_title,
                    "BPM": f"{detected_bpm:.1f}",
                    "Key": detected_key,
                    "File": os.path.basename(final_filename)
                })
                st.session_state.last_downloaded = final_filename

                st.success("Beat processed successfully!")
                c1, c2 = st.columns(2)
                with c1: st.metric("🎵 Tempo", f"{detected_bpm:.1f} BPM")
                with c2: st.metric("🎹 Key", detected_key)
                st.info(f"Saved: `{os.path.basename(final_filename)}`")

with tab_stems:
    st.subheader("🧬 AI Stem Splitter (Powered by Demucs)")
    st.markdown("Isolate vocals, drums, bass, or other instrumentation directly on your machine.")
    
    stem_target_file = st.text_input("Audio file path to split (defaults to last downloaded):", 
                                       value=st.session_state.get("last_downloaded", ""))
    
    stems_count = st.radio("Separation Model:", ["4 Stems (Drums, Bass, Vocals, Other)", "2 Stems (Vocals / Instrumental)"])
    
    if st.button("🚀 Run AI Stem Separation"):
        if stem_target_file and os.path.exists(stem_target_file):
            with st.spinner("AI is separating stems... (This may take a minute)"):
                output_dir = os.path.expanduser("~/Downloads/Stems")
                os.makedirs(output_dir, exist_ok=True)
                
                model_flag = "htdemucs" if "4 Stems" in stems_count else "htdemucs_ft"
                cmd = ["python3", "-m", "demucs.separate", "-n", model_flag, "-o", output_dir, stem_target_file]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success(f"Stems successfully extracted to `Downloads/Stems/{model_flag}/`!")
                else:
                    st.error(f"Stem separation error: {result.stderr}")
        else:
            st.warning("Please provide a valid audio file path.")

with tab_trim:
    st.subheader("✂️ Precision Loop Trimmer")
    target_file = st.text_input("File path to trim (defaults to last downloaded):", 
                                value=st.session_state.get("last_downloaded", ""))
    
    if target_file and os.path.exists(target_file):
        try:
            y_full, sr_full = librosa.load(target_file, sr=None)
            duration = librosa.get_duration(y=y_full, sr=sr_full)
            
            st.markdown(f"**Total Duration:** {duration:.2f} seconds")
            
            start_time, end_time = st.slider(
                "Select Loop Region (Seconds):",
                0.0, float(duration), (0.0, min(30.0, float(duration))),
                step=0.1
            )
            
            if st.button("💾 Export Trimmed Loop"):
                start_sample = int(start_time * sr_full)
                end_sample = int(end_time * sr_full)
                y_trimmed = y_full[start_sample:end_sample]
                
                base, ext = os.path.splitext(target_file)
                trimmed_filename = f"{base}_loop_{int(start_time)}s_to_{int(end_time)}s{ext}"
                sf.write(trimmed_filename, y_trimmed, sr_full)
                st.success(f"Trimmed loop exported successfully!")
                st.info(f"Saved as: `{os.path.basename(trimmed_filename)}`")
        except Exception as ex:
            st.warning(f"Could not load file for trimming: {ex}")
    else:
        st.markdown("*Process a beat in the download tab first, or paste a valid file path above.*")

with tab_history:
    st.subheader("Recent Session Crate")
    if st.session_state.history:
        st.table(st.session_state.history)
    else:
        st.markdown("*No beats processed yet.*")
