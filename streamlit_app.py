import streamlit as st
from groq import Groq
import json
import os
from io import BytesIO
import requests
import pydub
import re
import tempfile
# from md2pdf.core import md2pdf

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
x-rapidapi-key = os.environ.get("x-rapidapi-key", None)

if 'api_key' not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY

if 'groq' not in st.session_state:
    if GROQ_API_KEY:
        st.session_state.groq = Groq()


if 'x-rapidapi-key' not in st.session_state:
    st.session_state.api_key = x-rapidapi-key
	
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False

if 'button_text' not in st.session_state:
    st.session_state.button_text = "Generate Notes"

if 'statistics_text' not in st.session_state:
    st.session_state.statistics_text = ""

st.set_page_config(
    page_title="Groqnotes",
    page_icon="🗒️",
)
class GenerationStatistics:
    def __init__(self, input_time=0, output_time=0, input_tokens=0, output_tokens=0, total_time=0, model_name="llama3-8b-8192"):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time # Sum of queue, prompt (input), and completion (output) times
        self.model_name = model_name

    def get_input_speed(self):
        """ 
        Tokens per second calculation for input
        """
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0
    
    def get_output_speed(self):
        """ 
        Tokens per second calculation for output
        """
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0
    
    def add(self, other):
        """
        Add statistics from another GenerationStatistics object to this one.
        """
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")
        
        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        return (f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s  Model: {self.model_name}\n\n"
                f"| Metric          | Input          | Output          | Total          |\n"
                f"|-----------------|----------------|-----------------|----------------|\n"
                f"| Speed (T/s)     | {self.get_input_speed():.2f}            | {self.get_output_speed():.2f}            | {(self.input_tokens + self.output_tokens) / self.total_time if self.total_time != 0 else 0:.2f}            |\n"
                f"| Tokens          | {self.input_tokens}            | {self.output_tokens}            | {self.input_tokens + self.output_tokens}            |\n"
                f"| Inference Time (s) | {self.input_time:.2f}            | {self.output_time:.2f}            | {self.total_time:.2f}            |")

class NoteSection:
    def __init__(self, structure, transcript):
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}

        st.markdown("## Raw transcript:")
        st.markdown(transcript)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def return_existing_contents(self, level=1) -> str:
        existing_content = ""
        for title, content in self.structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                existing_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                existing_content += self.get_markdown_content(content, level + 1)
        return existing_content

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
        
        for title, content in structure.items():
            if self.contents[title].strip():  # Only display title if there is content
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        """
        Returns the markdown styled pure string with the contents.
        """
        if structure is None:
            structure = self.structure
        
        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content

def create_markdown_file(content: str) -> BytesIO:
    """
    Create a Markdown file from the provided content.
    """
    markdown_file = BytesIO()
    markdown_file.write(content.encode('utf-8'))
    markdown_file.seek(0)
    return markdown_file

# def create_pdf_file(content: str):
#     """
#     Create a PDF file from the provided content.
#     """
#     pdf_buffer = BytesIO()
#     md2pdf(pdf_buffer, md_content=content)
#     pdf_buffer.seek(0)
#     return pdf_buffer

def transcribe_audio(audio_file_path):
    """
    Transcribes audio using Groq's Whisper API.
    """
    if st.session_state.groq is None:
        raise ValueError("Groq client is not initialized")
    
    with open(audio_file_path, "rb") as audio_file:
        transcription = st.session_state.groq.audio.transcriptions.create(
          file=audio_file,
          model="whisper-large-v3",
          prompt="",
          response_format="json",
          language="zh",
          temperature=0.0 
        )

    results = transcription.text
    return results


def generate_notes_structure(transcript: str):
    """
    Returns notes structure content as well as total tokens and total time for generation.
    """

    shot_example = """{
        "Introduction": "Introduction to the AMA session, including the topic of Groq scaling architecture and the panelists",
        "Panelist Introductions": "Brief introductions from Igor, Andrew, and Omar, covering their backgrounds and roles at Groq",
        "Groq Scaling Architecture Overview": "High-level overview of Groq's scaling architecture, covering hardware, software, and cloud components",
        "Hardware Perspective": "Igor's overview of Groq's hardware approach, using an analogy of city traffic management to explain the traditional compute approach and Groq's innovative approach",
        "Traditional Compute": "Description of traditional compute approach, including asynchronous nature, queues, and poor utilization of infrastructure",
        "Groq's Approach": "Description of Groq's approach, including pre-orchestrated movement of data, low latency, high energy efficiency, and high utilization of resources",
        "Hardware Implementation": "Igor's explanation of the hardware implementation, including a comparison of GPU and LPU architectures"
        }"""
    completion = st.session_state.groq.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "Write in JSON format:\n\n{\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\"}"
            },
            {
                "role": "user",
                "content": f"### 抄本 {transcript}\n\n### 示例\n\n{shot_example}### 指令\n\n为上述转录的音频创建一个全面的中文笔记结构。部分标题和内容描述必须全面。质量重于数量。"
            }
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name="llama3-70b-8192")

    return statistics_to_return, completion.choices[0].message.content

def generate_section(transcript: str, existing_notes: str, section: str):
    stream = st.session_state.groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "你是一位专业的作家。基于提供的转录内容，为所提供的部分生成一份全面的中文笔记。不要重复之前部分的任何内容。"
            },
            {
                "role": "user",
                "content": f"### 转录内容：\n\n{transcript}\n\n### 已经有的笔记内容：\n\n{existing_notes}\n\n### 指令要求：\n\n 根据转录内容，仅为此部分生成全面的中文笔记：: \n\n{section}"
            }
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            statistics_to_return = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name="llama3-8b-8192")
            yield statistics_to_return

# Initialize
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False

if 'button_text' not in st.session_state:
    st.session_state.button_text = "Generate Notes"

if 'statistics_text' not in st.session_state:
    st.session_state.statistics_text = ""

st.write("""
# Douyin Note：根据抖音视频创建结构化笔记 🗒️⚡
""")

def disable():
    st.session_state.button_disabled = True

def enable():
    st.session_state.button_disabled = False

def empty_st():
    st.empty()

def get_audio_url_from_video(video_url):
    """
    Get audio URL from the video URL using a third-party API.
    """
    api_url = "https://auto-download-all-in-one.p.rapidapi.com/v1/social/autolink"
    payload = {"url": video_url}
    headers = {
	"x-rapidapi-key": x-rapidapi-key,
	"x-rapidapi-host": "auto-download-all-in-one.p.rapidapi.com",
	"Content-Type": "application/json"
    }
    
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        
        medias = response.json().get("medias", [])
        
        # audio_url = next((media["url"] for media in medias["medias"] if media["type"] == "audio"), None)
        audio_url = next((media["url"] for media in medias if media["type"] == "audio"), None)
        if not audio_url:
            raise ValueError("No audio URL found in the response")
        return audio_url
    else:
        raise ValueError("Failed to get audio URL from video URL")

def download_audio(audio_url):
    """
    Download audio from the provided URL and save it as a file.
    """
    response = requests.get(audio_url)
    if response.status_code == 200:
        audio_file = BytesIO(response.content)
        return audio_file
    else:
        raise ValueError("Failed to download audio from URL")

def convert_audio_format(audio_file, target_format='mp3'):
    """
    Convert audio file to a supported format using pydub.
    """
    audio = pydub.AudioSegment.from_file(audio_file)
    output = BytesIO()
    audio.export(output, format=target_format)
    output.seek(0)
    return output


def extract_first_url(text):
    # 正则表达式匹配 URL
    url_pattern = r'(https?://[^\s]+)'
    match = re.search(url_pattern, text)
    if match:
        return match.group(0)
    else:
        return None


def save_to_temp_file(audio_data, target_format='mp3'):
    """
    Save BytesIO audio data to a temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{target_format}') as temp_file:
        temp_file.write(audio_data.read())
        temp_file_path = temp_file.name
    return temp_file_path



try:
    # with st.sidebar:
    #     audio_files = {
    #         "Transformers Explained by Google Cloud Tech": {
    #             "file_path": "./assets/audio/transformers_explained.m4a",
    #             "youtube_link": "https://www.youtube.com/watch?v=SZorAJ4I-sA"
    #         },
         
    #     }

    #     st.write(f"# 🗒️ Groqnotes \n## Generate notes from audio in seconds using Groq, Whisper, and Llama3")
    #     st.markdown(f"[Github Repository](https://github.com/bklieger/groqnotes)\n\nAs with all generative AI, content may include inaccurate or placeholder information. Groqnotes is in beta and all feedback is welcome!")

    #     st.write(f"---")

    #     st.write(f"# Sample Audio Files")

        # for audio_name, audio_info in audio_files.items():

        #     st.write(f"### {audio_name}")
            
        #     # Read audio file as binary
        #     with open(audio_info['file_path'], 'rb') as audio_file:
        #         audio_bytes = audio_file.read()

        #     # Create download button
        #     st.download_button(
        #         label=f"Download audio",
        #         data=audio_bytes,
        #         file_name=audio_info['file_path'],
        #         mime='audio/m4a'
        #     )
            
        #     st.markdown(f"[Credit Youtube Link]({audio_info['youtube_link']})")
        #     st.write(f"\n\n")


    if st.button('End Generation and Download Notes'):
        if "notes" in st.session_state:

            # Create markdown file
            markdown_file = create_markdown_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download Text',
                data=markdown_file,
                file_name='generated_notes.txt',
                mime='text/plain'
            )

            # Create pdf file (styled)
            # pdf_file = create_pdf_file(st.session_state.notes.get_markdown_content())
            # st.download_button(
            #     label='Download PDF',
            #     data=pdf_file,
            #     file_name='generated_notes.pdf',
            #     mime='application/pdf'
            # )
        else:
            raise ValueError("Please generate content first before downloading the notes.")

    with st.form("groqform"):
        # audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"]) # TODO: Add a max size
        audio_file = ""
        video_url = st.text_input("输入视频地址")  # 新增视频地址输入框
        
        video_url = extract_first_url(video_url)
        
        st.session_state.statistics_text = video_url

        # Generate button
        submitted = st.form_submit_button(st.session_state.button_text, on_click=disable, disabled=st.session_state.button_disabled)
        
        # Statistics
        placeholder = st.empty()
        def display_statistics():
            with placeholder.container():
                if st.session_state.statistics_text:
                    if "Transcribing audio in background22" not in st.session_state.statistics_text:
                        st.markdown(st.session_state.statistics_text + "\n\n---\n")  # Format with line if showing statistics
                    else:
                        st.markdown(st.session_state.statistics_text)
                else:
                    placeholder.empty()

        if submitted:
            if audio_file is None and not video_url:
                raise ValueError("Please upload an audio file or input a video URL")

            st.session_state.button_disabled = True
            st.session_state.statistics_text = "正在后台转录视频内容，请等待几秒钟 ...."  # Show temporary message before transcription is generated and statistics show
            display_statistics()

            if audio_file:
                transcription_text = transcribe_audio(audio_file)
            elif video_url:
                audio_url = get_audio_url_from_video(video_url)
                audio_file = download_audio(audio_url)
                audio_file = convert_audio_format(audio_file)  # Convert to mp3 if necessary
                audio_file_path = save_to_temp_file(audio_file, target_format='mp3')
                transcription_text = transcribe_audio(audio_file_path)

            large_model_generation_statistics, notes_structure = generate_notes_structure(transcription_text)
            print("Structure: ", notes_structure)

            total_generation_statistics = GenerationStatistics(model_name="llama3-8b-8192")

            try:
                notes_structure_json = json.loads(notes_structure)
                notes = NoteSection(structure=notes_structure_json, transcript=transcription_text)
                
                if 'notes' not in st.session_state:
                    st.session_state.notes = notes

                st.session_state.notes.display_structure()

                def stream_section_content(sections):
                    for title, content in sections.items():
                        if isinstance(content, str):
                            content_stream = generate_section(transcript=transcription_text, existing_notes=notes.return_existing_contents(), section=(title + ": " + content))
                            for chunk in content_stream:
                                # Check if GenerationStatistics data is returned instead of str tokens
                                chunk_data = chunk
                                if type(chunk_data) == GenerationStatistics:
                                    total_generation_statistics.add(chunk_data)
                                    
                                    st.session_state.statistics_text = str(total_generation_statistics)
                                    display_statistics()
                                elif chunk is not None:
                                    st.session_state.notes.update_content(title, chunk)
                        elif isinstance(content, dict):
                            stream_section_content(content)

                stream_section_content(notes_structure_json)
            
            except json.JSONDecodeError:
                st.error("Failed to decode the notes structure. Please try again.")

            enable()

except Exception as e:
    st.session_state.button_disabled = False
    st.error(e)

    if st.button("Clear"):
        st.rerun()
