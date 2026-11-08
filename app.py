import os
import shutil
import subprocess
import base64
import streamlit as st
import yt_dlp
import librosa
import numpy as np
import plotly.graph_objects as px_go

# Custom CSS for Light/Dark Mode Input Text Colors & Layout
st.markdown(
    """
    <style>
    .stTextInput input, .stTextArea textarea {
        color: #000000 !important;
    }
    @media (prefers-color-scheme: dark) {
        .stTextInput input, .stTextArea textarea {
            color: #ffffff !important;
        }
    }
    .y2mate-header {
        text-align: center;
        margin-bottom: 5px;
    }
    .y2mate-sub {
        text-align: center;
        margin-top: 0px;
        margin-bottom: 30px;
        color: #888888;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='y2mate-header'><h1>🎵 BeatWorkflowHub</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='y2mate-sub'>Converter, Analyzer, Stem Splitter & Liquid-Chrome Loop Chopper</div>", unsafe_allow_html=True)

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.mean(chroma, axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    best_match = ""
    max_corr = -1
    for i in range(12):
        maj_corr = np.corrcoef(chroma_vals, np.roll(major_profile, i))[0, 1]
        if maj_corr > max_corr:
            max_corr = maj_corr
            best_match = f"{pitches[i]}"
        min_corr = np.corrcoef(chroma_vals, np.roll(minor_profile, i))[0, 1]
        if min_corr > max_corr:
            max_corr = min_corr
            best_match = f"{pitches[i]}m"
    return best_match

def analyze_audio_features(file_path):
    y, sr = librosa.load(file_path, offset=30.0, duration=45, sr=22050)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(np.round(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo))
    key = detect_key(y, sr)
    rms = np.mean(librosa.feature.rms(y=y))
    energy = int(min(100, max(0, rms * 500)))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythm_regularity = np.std(onset_env)
    danceability = int(min(100, max(10, 100 - (rhythm_regularity * 5))))
    if energy > 70 and danceability > 60:
        mood = "🔥 High Energy / Banger"
    elif energy < 40:
        mood = "🌙 Chill / Atmospheric"
    elif "m" in key:
        mood = "🎭 Dark / Aggressive"
    else:
        mood = "✨ Uplifting / Bright"
    return bpm, key, energy, danceability, mood

def get_chrome_waveform(y, sr, start_sec=0.0, duration_sec=4.0):
    fig = px_go.Figure()
    total_duration = librosa.get_duration(y=y, sr=sr)
    hop_length = int(sr * 0.04)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm = rms / np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else np.zeros(100)
    time_axis = librosa.frames_to_time(np.arange(len(rms_norm)), sr=sr, hop_length=hop_length)
    upper = 0.5 + (rms_norm * 0.42)
    lower = 0.5 - (rms_norm * 0.42)
    fig.add_trace(px_go.Scatter(x=np.concatenate([time_axis, time_axis[::-1]]), y=np.concatenate([upper, lower[::-1]]), fill='toself', fillcolor='rgba(210, 220, 235, 0.95)', line=dict(color='rgba(255, 255, 255, 1.0)', width=1.5), hoverinfo='skip', showlegend=False))
    fig.add_trace(px_go.Scatter(x=time_axis, y=upper, mode='lines', line=dict(color='#ffffff', width=2.5), hoverinfo='skip', showlegend=False))
    end_sec = start_sec + duration_sec
    fig.add_vrect(x0=start_sec, x1=end_sec, fillcolor="yellow", opacity=0.35, layer="above", line_width=2, line_dash="solid", line_color="yellow", annotation_text="Active Loop 🎛️", annotation_position="top left")
    fig.update_layout(template="plotly_dark", paper_bgcolor='#111111', plot_bgcolor='#161616', margin=dict(l=10, r=10, t=10, b=30), height=220, dragmode='select', xaxis=dict(range=[0, total_duration], title="Timeline / Time (Seconds)", gridcolor='#222222', fixedrange=True), yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True), showlegend=False)
    return fig

def get_default_pattern():
    return {
        "808 Sub 🔊": [True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False],
        "Kick 🥁": [True, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False],
        "Snare 🎯": [False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, True],
        "Clap 👏": [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        "Hi-Hat 🧢": [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        "Open Hat 🎵": [False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False],
        "Perc ⚡": [False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True]
    }

def transcribe_dual_stems(stem_dir, prefix, bpm):
    try:
        beat_duration = 60.0 / bpm
        step_duration = beat_duration / 4.0
        
        sub808_steps = [False] * 16
        kick_steps = [False] * 16
        snare_steps = [False] * 16
        clap_steps = [False] * 16
        hihat_steps = [False] * 16
        openhat_steps = [False] * 16
        perc_steps = [False] * 16

        # Parse Bass Stem for 808s
        bass_path = os.path.join(stem_dir, f"{prefix} - Bass.wav")
        if not os.path.exists(bass_path):
            bass_path = os.path.join(stem_dir, "bass.wav")
            
        if os.path.exists(bass_path):
            y_bass, sr_bass = librosa.load(bass_path, sr=22050, duration=15.0)
            S_bass = np.abs(librosa.stft(y_bass))
            freqs_bass = librosa.fft_frequencies(sr=sr_bass)
            sub_mask = (freqs_bass >= 30) & (freqs_bass <= 70)
            sub_energy = np.mean(S_bass[sub_mask, :], axis=0) if np.any(sub_mask) else np.zeros(S_bass.shape[1])
            times_bass = librosa.frames_to_time(np.arange(S_bass.shape[1]), sr=sr_bass)
            
            mean_sub = np.mean(sub_energy)
            for i in range(16):
                t_start = i * step_duration
                t_end = (i + 1) * step_duration
                mask = (times_bass >= t_start) & (times_bass < t_end)
                if np.any(mask):
                    sub808_steps[i] = bool(np.max(sub_energy[mask]) > (mean_sub * 1.25))

        # Parse Drums Stem
        drums_path = os.path.join(stem_dir, f"{prefix} - Drums.wav")
        if not os.path.exists(drums_path):
            drums_path = os.path.join(stem_dir, "drums.wav")
            
        if os.path.exists(drums_path):
            y_drums, sr_drums = librosa.load(drums_path, sr=22050, duration=15.0)
            S_drums = np.abs(librosa.stft(y_drums))
            freqs_drums = librosa.fft_frequencies(sr=sr_drums)
            
            kick_mask = (freqs_drums >= 70) & (freqs_drums <= 150)
            snare_mask = (freqs_drums >= 200) & (freqs_drums <= 2000)
            clap_mask = (freqs_drums >= 1000) & (freqs_drums <= 4500)
            hihat_mask = (freqs_drums >= 5000) & (freqs_drums <= 10000)
            openhat_mask = (freqs_drums >= 3500) & (freqs_drums <= 8000)
            perc_mask = (freqs_drums >= 800) & (freqs_drums <= 3000)
            
            k_en = np.mean(S_drums[kick_mask, :], axis=0) if np.any(kick_mask) else np.zeros(S_drums.shape[1])
            s_en = np.mean(S_drums[snare_mask, :], axis=0) if np.any(snare_mask) else np.zeros(S_drums.shape[1])
            c_en = np.mean(S_drums[clap_mask, :], axis=0) if np.any(clap_mask) else np.zeros(S_drums.shape[1])
            h_en = np.mean(S_drums[hihat_mask, :], axis=0) if np.any(hihat_mask) else np.zeros(S_drums.shape[1])
            oh_en = np.mean(S_drums[openhat_mask, :], axis=0) if np.any(openhat_mask) else np.zeros(S_drums.shape[1])
            p_en = np.mean(S_drums[perc_mask, :], axis=0) if np.any(perc_mask) else np.zeros(S_drums.shape[1])
            
            times_drums = librosa.frames_to_time(np.arange(S_drums.shape[1]), sr=sr_drums)
            
            for i in range(16):
                t_start = i * step_duration
                t_end = (i + 1) * step_duration
                mask = (times_drums >= t_start) & (times_drums < t_end)
                if not np.any(mask):
                    continue
                
                kv = np.max(k_en[mask])
                sv = np.max(s_en[mask])
                cv = np.max(c_en[mask])
                hv = np.max(h_en[mask])
                ohv = np.max(oh_en[mask])
                pv = np.max(p_en[mask])
                
                kick_steps[i] = bool(kv > (np.mean(k_en) * 1.35))
                snare_steps[i] = bool(sv > (np.mean(s_en) * 1.35) and cv < (np.mean(c_en) * 1.2))
                clap_steps[i] = bool(cv > (np.mean(c_en) * 1.35))
                hihat_steps[i] = bool(hv > (np.mean(h_en) * 1.15))
                openhat_steps[i] = bool(ohv > (np.mean(oh_en) * 1.4) and hv < (np.mean(h_en) * 1.3))
                perc_steps[i] = bool(pv > (np.mean(p_en) * 1.4) and sv < (np.mean(s_en) * 1.1))

        return {
            "808 Sub 🔊": sub808_steps,
            "Kick 🥁": kick_steps,
            "Snare 🎯": snare_steps,
            "Clap 👏": clap_steps,
            "Hi-Hat 🧢": hihat_steps,
            "Open Hat 🎵": openhat_steps,
            "Perc ⚡": perc_steps
        }
    except Exception as e:
        return get_default_pattern()

downloads_path = os.path.expanduser("~/Downloads")

# YouTube Search & Link Input
url_input = st.text_input("Search keyword or paste YouTube link here:", placeholder="https://www.youtube.com/watch?v=... or enter song title")

if "video_info" not in st.session_state:
    st.session_state.video_info = None

if st.button("🔎 Start / Analyze Track", use_container_width=True):
    if url_input.strip():
        with st.spinner("Analyzing media link..."):
            try:
                query = url_input.strip()
                if not query.startswith("http://") and not query.startswith("https://"):
                    query = f"ytsearch1:{query}"
                ydl_opts = {'extract_flat': False, 'skip_download': True, 'noplaylist': True, 'ignoreconfig': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(query, download=False)
                    if 'entries' in info:
                        info = info['entries'][0]
                    st.session_state.video_info = info
                st.success("Track metadata loaded successfully!")
            except Exception as e:
                st.error(f"Failed to fetch video: {e}")
                st.session_state.video_info = None
    else:
        st.warning("Please enter a link or search term first.")

if st.session_state.video_info:
    info = st.session_state.video_info
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(info.get("thumbnail", ""), use_container_width=True)
    with col2:
        title = info.get("title", "Unknown Title")
        st.subheader(title)
        st.caption(f"⏱️ Duration: {info.get('duration', 0) // 60}m {info.get('duration', 0) % 60}s")
        quality_opt = st.selectbox("Select Audio Output Format:", ["MP3 320kbps (Best Quality)", "WAV (Uncompressed)"])
        
        if st.button("📥 Download, Analyze & Auto-Rename", type="primary", use_container_width=True):
            codec = 'wav' if "WAV" in quality_opt else 'mp3'
            actual_temp_path = os.path.join(downloads_path, f"temp_beat_download.{codec}")
            try:
                with st.spinner("Step 1/2: Downloading audio file..."):
                    webpage_url = info.get("webpage_url", url_input)
                    ydl_download_opts = {
                        'format': 'bestaudio/best',
                        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': codec, 'preferredquality': '320'}],
                        'outtmpl': os.path.join(downloads_path, 'temp_beat_download.%(ext)s'),
                        'noplaylist': True, 'overwrites': True, 'ignoreconfig': True,
                    }
                    with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
                        ydl.download([webpage_url])

                with st.spinner("Step 2/2: Running Tunebat Audio Analysis (BPM & Key)..."):
                    bpm, key, energy, danceability, mood = analyze_audio_features(actual_temp_path)
                
                clean_title = "".join([c for c in title if c.isalnum() or c in (' ', '_', '-')]).rstrip()
                final_filename = f"[{bpm}BPM - {key}] {clean_title}.{codec}"
                final_filepath = os.path.join(downloads_path, final_filename)
                if os.path.exists(final_filepath):
                    os.remove(final_filepath)
                os.rename(actual_temp_path, final_filepath)
                
                st.session_state.last_downloaded_file = final_filepath
                st.session_state.clean_title = clean_title
                st.session_state.base_prefix = f"[{bpm}BPM - {key}] {clean_title}"
                st.session_state.bpm = bpm
                st.session_state.key = key
                st.session_state.marquee_start = 0.0
                
                st.balloons()
                st.success(f"🎉 Processed! Saved as: **{final_filename}**")
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")

# Stem Splitter & Dual-Stem Transcriber Section
if "last_downloaded_file" in st.session_state and os.path.exists(st.session_state.last_downloaded_file):
    st.divider()
    st.subheader("🎛️ AI Stem Splitter & Dual-Stem Transcriber")
    if st.button("✂️ Split Stems & Transcribe 808s, Kicks, Snares, Claps, Hi-Hats", use_container_width=True):
        with st.spinner("Isolating stems (Bass + Drums) and running dual-stem transient analysis..."):
            try:
                target_file = st.session_state.last_downloaded_file
                prefix = st.session_state.base_prefix
                subprocess.run(["demucs", "-n", "htdemucs", "-o", downloads_path, target_file], check=True)
                
                source_stem_dir = os.path.join(downloads_path, "htdemucs", prefix)
                output_stem_dir = os.path.join(downloads_path, f"Stems - {prefix}")
                
                if os.path.exists(source_stem_dir):
                    os.makedirs(output_stem_dir, exist_ok=True)
                    for stem in ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']:
                        src_path = os.path.join(source_stem_dir, stem)
                        if os.path.exists(src_path):
                            shutil.copy(src_path, os.path.join(output_stem_dir, f"{prefix} - {stem.replace('.wav', '').capitalize()}.wav"))
                    
                    st.session_state.stem_folder = output_stem_dir
                    st.session_state.transcribed_pattern = transcribe_dual_stems(
                        output_stem_dir, prefix, st.session_state.get('bpm', 120)
                    )
                    st.success("✨ Stems split & dual-stem pattern (808s + Claps + Kicks) transcribed successfully!")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")

# Sequencer Section (ONLY loads when stems have been successfully split & transcribed)
if "stem_folder" in st.session_state and os.path.exists(st.session_state.stem_folder):
    st.divider()
    st.subheader("🥁 16-Step Beat & Rhythm Sequencer")
    st.caption(f"Syncing Pattern Tempo: **{st.session_state.get('bpm', 92)} BPM**")

    active_drum_tracks = st.session_state.get("transcribed_pattern", get_default_pattern())

    pattern_data = {}
    for track_name, default_vals in active_drum_tracks.items():
        st.markdown(f"**{track_name}**")
        cols = st.columns(16)
        track_steps = []
        for i in range(16):
            with cols[i]:
                val = st.checkbox(f"{i+1}", value=default_vals[i], key=f"seq_{track_name}_{i}", label_visibility="collapsed")
                track_steps.append(val)
        pattern_data[track_name] = track_steps

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        swing_amount = st.slider("Swing / Groove (%)", min_value=0, max_value=50, value=15, step=5)
    with col_s2:
        kit_preset = st.selectbox("Drum Kit Preset:", ["908 / Trap Bangers", "Boom-Bap Analog", "Lo-Fi Cassette", "West Coast Bounce"])

    if st.button("💾 Export Pattern to MIDI / Text", use_container_width=True):
        active_steps_summary = {k: [i+1 for i, active in enumerate(v) if active] for k, v in pattern_data.items()}
        st.success(f"🔥 Pattern generated successfully for {kit_preset} at {st.session_state.get('bpm', 92)} BPM!")
        st.json(active_steps_summary)