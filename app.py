import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def main():
    st.logo("images/logo.png")
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Home", "Preprocessing", "Testing"])

    if page == "Home":
        show_home()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title("Pengolahan Data TBC dengan Metode Decision Tree")

    # Explain what is Decision Tree
    st.header("Apa itu Decision Tree?")
    st.write("Decision Tree adalah salah satu metode klasifikasi yang paling populer dalam machine learning. Decision Tree memodelkan keputusan dalam bentuk struktur pohon yang terdiri dari node dan leaf. Setiap node dalam pohon mewakili fitur atau atribut, sedangkan setiap leaf mewakili label atau kelas.")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data TBC dengan menggunakan metode Decision Tree.")

    # Explain the data
    st.header("Data")
    st.write("Data yang digunakan diambil dari dosen mata kuliah yang berisi informasi terkait TBC.")

    # Explain the process of Decision Tree
    st.header("Tahapan Proses Decision Tree")
    st.write("1. **Pengumpulan Data**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Pemodelan**")
    st.write("4. **Evaluasi Model**")
    st.write("5. **Prediksi**")

def show_preprocessing():
    st.title("Preprocessing Data")
    df = pd.read_csv("Data_TB.csv")
    
    st.write("Data yang tersedia:")
    st.write(df.head())
    
    st.write("## Preprocessing")
    df = df.rename({'LOKASI ANATOMI (target/output)': 'LOKASI ANATOMI'}, axis=1)


    # --------------- Missing value -----------------
    st.write("### Missing Value")

    tcm = len(df[df['HASIL TCM'] == 'Tidak dilakukan'])
    toraks = len(df[df['FOTO TORAKS'] == 'Tidak dilakukan'])
    hiv = len(df[df['STATUS HIV'] == 'Tidak diketahui'])
    diabet = len(df[df['RIWAYAT DIABETES'] == 'Tidak diketahui'])

    st.write("\nData dengan 'HASIL TCM' == 'Tidak dilakukan':")
    st.write(tcm)

    st.write("\nData dengan 'FOTO TORAKS' == 'Tidak dilakukan':")
    st.write(toraks)

    st.write("\nData dengan 'STATUS HIV' == 'Tidak diketahui':")
    st.write(hiv)

    st.write("\nData dengan 'RIWAYAT DIABETES' == 'Tidak diketahui':")
    st.write(diabet)

    # --------------- Drop attribute -----------------
    st.write("### Drop Atribut")
    st.write("Menghapus kolom 'Umur' dan 'kecamatan' karena tidak diperlukan")
    df = df.drop(["KECAMATAN"], axis=1)
    df = df.drop(["UMUR"], axis=1)

    # --------------- Pengisian Data -----------------
    st.write("### Pengisian Data")
    st.write("Mengisi nilai yang hilang")
    df['FOTO TORAKS'] = df['FOTO TORAKS'].replace('Tidak dilakukan', 'Negatif')
    df['STATUS HIV'] = df['STATUS HIV'].replace('Tidak diketahui', 'Negatif')
    df['RIWAYAT DIABETES'] = df['RIWAYAT DIABETES'].replace('Tidak diketahui', 'Tidak')
    df['HASIL TCM'] = df['HASIL TCM'].replace('Tidak dilakukan', 'Rif Sensitif')
    st.write(df.head())


    col1, col2, col3 = st.columns(3,vertical_alignment='bottom')

    # --------------- Sebelum mapping -----------------
    with col1:
        st.header("Sebelum Mapping")
        st.write("JENIS KELAMIN:", df['JENIS KELAMIN'].unique())
        st.write("FOTO TORAKS:", df['FOTO TORAKS'].unique())
        st.write("STATUS HIV:", df['STATUS HIV'].unique())
        st.write("RIWAYAT DIABETES:", df['RIWAYAT DIABETES'].unique())
        st.write("HASIL TCM:", df['HASIL TCM'].unique())
        st.write("LOKASI ANATOMI:", df['LOKASI ANATOMI'].unique())

    with col2:
        st.header("Setelah Mapping")
        # --------------- Mapping Process -----------------
        df['JENIS KELAMIN'] = df['JENIS KELAMIN'].map({'P': 0, 'L': 1})
        df['FOTO TORAKS'] = df['FOTO TORAKS'].map({'Negatif': 0, 'Positif': 1})
        df['STATUS HIV'] = df['STATUS HIV'].map({'Negatif': 0, 'Positif': 1})
        df['RIWAYAT DIABETES'] = df['RIWAYAT DIABETES'].map({'Tidak': 0, 'Ya': 1})
        df['HASIL TCM'] = df['HASIL TCM'].map({'Negatif': 1, 'Rif Sensitif': 0, 'Rif resisten': 2})
        df['LOKASI ANATOMI'] = df['LOKASI ANATOMI'].map({'Paru': 0, 'Ekstra paru': 1})

        # --------------- Setelah mapping -----------------
        st.write("JENIS KELAMIN:", df['JENIS KELAMIN'].unique())
        st.write("FOTO TORAKS:", df['FOTO TORAKS'].unique())
        st.write("STATUS HIV:", df['STATUS HIV'].unique())
        st.write("RIWAYAT DIABETES:", df['RIWAYAT DIABETES'].unique())
        st.write("HASIL TCM:", df['HASIL TCM'].unique())
        st.write("LOKASI ANATOMI:", df['LOKASI ANATOMI'].unique())

    with col3:
        st.header("Tipe Data")
        # --------------- Type Data -----------------
        st.write(df.dtypes)



    # --------------- Final data -----------------
    st.write("### Data Akhir")
    st.write(df.head())

    # st.write("### Normalisasi data")
    # scaler = StandardScaler()
    # df[df.columns] = scaler.fit_transform(df[df.columns])
    # st.write(df.head())
    
    st.session_state['preprocessed_data'] = df

def show_testing():
    st.title("Testing Model")
    
    if 'preprocessed_data' not in st.session_state:
        st.write("Silakan lakukan preprocessing data terlebih dahulu.")
        return
    
    df = st.session_state['preprocessed_data']
    
    st.write("Data yang telah dipreproses:")
    st.write(df.head())

    # Memisahkan fitur dan label
    # Misalkan kolom terakhir adalah label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Memastikan semua data fitur adalah numerik
    X = X.apply(pd.to_numeric, errors='coerce')
    y = y.astype(str)  # Mengkonversi label menjadi string jika diperlukan

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Membuat model Decision Tree dengan kriteria 'entropy' untuk C4.5
    clf = DecisionTreeClassifier(criterion='entropy')

    # Melatih model
    clf = clf.fit(X_train, y_train)

    # Memprediksi data uji
    y_pred = clf.predict(X_test)

    # Evaluasi model
    st.subheader("Akurasi")
    st.write(metrics.accuracy_score(y_test, y_pred))

    classification_report = metrics.classification_report(y_test, y_pred)
    st.subheader("Classification Report")
    st.text(classification_report)

    # Menampilkan Confusion Matrix sebagai tabel
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix_df = pd.DataFrame(confusion_matrix, 
                                    index=[f'Actual {i}' for i in range(len(confusion_matrix))], 
                                    columns=[f'Predicted {i}' for i in range(len(confusion_matrix))])
    st.subheader("Confusion Matrix")
    st.table(confusion_matrix_df)

    # Visualisasi Confusion Matrix
    # cm = metrics.confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # st.pyplot(plt)

    # Menampilkan pohon keputusan
    plt.figure(figsize=(25,12))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
    st.subheader("Pohon Keputusan")
    st.pyplot(plt)

if __name__ == "__main__":
    st.set_page_config(page_title="Decision Tree", page_icon="🌳")
    main()
