from customtkinter import *
import joblib
from PIL import Image, ImageTk
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from docx import Document
# from tkinter import filedialog
# nltk.download('wordnet')
# nltk.download('punkt')

factory = StemmerFactory()
stemmer = factory.create_stemmer()


app = CTk()
app.geometry("900x6000")


# background_image = Image.open("background.png")
# background_photo = ImageTk.PhotoImage(background_image)
''
# background_label = CTkLabel(master=app, image=background_photo)
# background_label.place(relwidth=1, relheight=1)


tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
classifier = joblib.load('classifier_model.pkl')


main_frame = CTkFrame(master=app, fg_color='#4EAC7D')
main_frame.place(relx=0.5, rely=0.5, anchor='center')


frame_3 = CTkFrame(master=main_frame, fg_color="#7D4EAC")
frame_3.grid(row=1, column=1, padx=20, pady=20)

label = CTkLabel(master=main_frame, text="Analisis Review",
                 font=("Arial Bold", 30), justify="left")
label.grid(row=0, column=1, padx=50, pady=24)

label = CTkLabel(master=frame_3, text="Ketik atau Upload File",
                 font=("Arial Bold", 20), justify="left")
label.pack(expand=True, pady=(30, 15))


hasil_entry = CTkTextbox(master=frame_3, width=400)
hasil_entry.pack(expand=True, pady=10, padx=20)


factory = StemmerFactory()
stemmer = factory.create_stemmer()

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()


url = 'https://raw.githubusercontent.com/nadyndyaa/Kamus-Alay/main/Kamus%20Alay.csv'
formal = pd.read_csv(url, sep=',')
singkatan_dict = formal.set_index('Alay')['Baik'].to_dict()

formal = pd.read_csv('KamusAlay+.csv', sep=';')
singkatan_dict = formal.set_index('NonFormal')['Formal'].to_dict()


def ganti_singkatan(teks):
    singkatan = re.findall(r'\b\w+\b', teks)
    for kata in singkatan:
        if kata in singkatan_dict:
            teks = re.sub(r'\b{}\b'.format(kata),
                          singkatan_dict[kata], teks, flags=re.IGNORECASE)
    return teks


def ganti_singkatan_(teks):
    singkatan = re.findall(r'\b\w+\b', teks)
    for kata in singkatan:
        if kata in singkatan_dict:
            teks = re.sub(r'\b{}\b'.format(kata),
                          singkatan_dict[kata], teks, flags=re.IGNORECASE)
    return teks


def _normalize_whitespace(text):
    corrected = str(text)
    corrected = re.sub(r"//t", r"\t", corrected)
    corrected = re.sub(r"( )\1+", r"\1", corrected)
    corrected = re.sub(r"(\n)\1+", r"\1", corrected)
    corrected = re.sub(r"(\r)\1+", r"\1", corrected)
    corrected = re.sub(r"(\t)\1+", r"\1", corrected)
    return corrected.strip(" ")


stopword = factory.create_stop_word_remover()


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def click_button():
    entered_text = hasil_entry.get("1.0", "end-1c")
    if not entered_text:
        hasil_entry.delete("1.0", "end")
        hasil_entry.insert("1.0", "Error: Masukkan kalimat terlebih dahulu.")
        return
    teks_bersih = entered_text.lower()
    teks_bersih = re.sub(r'(.)\1+', r'\1', teks_bersih)
    teks_bersih = ganti_singkatan(teks_bersih)
    teks_bersih = ganti_singkatan_(teks_bersih)
    teks_bersih = re.sub(r"\d+", " ", teks_bersih)
    teks_bersih = re.compile('[/(){}\[\]\|@,;]').sub('', teks_bersih)
    teks_bersih = re.compile('[^0-9a-z]').sub(' ', teks_bersih)
    teks_bersih = _normalize_whitespace(teks_bersih)
    teks_bersih = stopword.remove(teks_bersih)
    teks_bersih = lemmatize_text(teks_bersih)
    teks_bersih_ = stemmer.stem(teks_bersih)

    hasil_entry.delete("1.0", "end")
    hasil_entry.insert("1.0", entered_text)
    tfidf_data_baru = tfidf_vectorizer.transform([teks_bersih_])
    hasil_prediksi = classifier.predict(tfidf_data_baru)
    hasil.delete("1.0", "end")
    hasil.insert(
        "end", f"Hasil Prediksi:{hasil_prediksi[0]}")


def open_file():
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("Word Files", "*.docx"), ("All Files", "*.*")]
    )

    if file_path:
        doc = Document(file_path)
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

        hasil_entry.delete("1.0", "end")
        hasil_entry.insert("1.0", content)


button = CTkButton(master=frame_3, text="Prediksi", command=click_button)
button.pack(side='left', expand=True, pady=(30, 15), padx=10)


button2 = CTkButton(master=frame_3, text="Upload File",
                    command=open_file)
button2.pack(side="left", expand=True, pady=(30, 15), padx=10)

hasil = CTkTextbox(master=main_frame, height=20)

hasil.grid(row=3, column=1, padx=5, pady=5)

hasil_label = CTkLabel(master=main_frame, text='Hasil',
                       font=("Arial Bold", 20), justify="left")

hasil_label.grid(row=2, column=1, padx=5, pady=5)


app.mainloop()
