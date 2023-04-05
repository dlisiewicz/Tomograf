import numpy as np
import streamlit as st
import cv2
import math
from bresenham import bresenham
from datetime import date
import pydicom
from pydicom.data import get_testdata_files
import skimage.filters.rank
import re
from prettytable import PrettyTable
from skimage.filters.edges import convolve



# oblicza srednią wartość piksela między emiterem a dekoderem oraz punkty pomiędzy nimi
def avg_pixel(A, B, image):
    x1 = int(A[0] + image.shape[0] / 2)
    y1 = int(A[1] + image.shape[0] / 2)
    x2 = int(B[0] + image.shape[0] / 2)
    y2 = int(B[1] + image.shape[0] / 2)

    points = list(bresenham(x1, y1, x2, y2))
    result = []
    for point in points:
        result.append(image[int(point[0])][int(point[1])])
    return sum(result)/len(result), points


# zwraca listę pozycji [x,y] detektorow
def get_detector_positions(r, step, theta, number_of_detectors):
    result = []
    for i in range(number_of_detectors):
        angle = step + 180 - theta / 2 + i * (theta / (number_of_detectors - 1))
        result.append([r * math.cos(math.radians(angle)), r * math.sin(math.radians(angle))])
    return result


# normalizacja obrazu
def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def dicom(image, name, sex, birth, description):
    filename = get_testdata_files("CT_small.dcm")[0];
    ds = pydicom.dcmread(filename)

    image2 = np.asarray(image, dtype=np.uint16)
    ds.Rows = image2.shape[1]
    ds.Columns = image2.shape[0]
    ds.PixelData = image2.tobytes()

    ds.PatientName = name
    ds.PatientSex = sex
    ds.PatientBirthDate = birth

    ds.StudyDate = date.today().strftime("%d.%m.%Y")
    ds.StudyDescription = description

    ds.save_as("Tomograf.dcm")


def filter(sin_row):
    size = 50
    kernel = []
    for k in range(-size // 2, size // 2):
        if k == 0:
            kernel.append(1)
        else:
            if k % 2 == 0:
                kernel.append(0)
            if k % 2 == 1:
                kernel.append((-4 / (math.pi ** 2)) / k ** 2)

    filtered = np.convolve(sin_row, kernel, mode='same')  # splot
    return filtered


def square(image):
    max_len = max(image.shape[0], image.shape[1])
    result = np.zeros((max_len, max_len))
    ax, ay = (max_len - image.shape[1]) // 2, (max_len - image.shape[0]) // 2
    result[ay:image.shape[0] + ay, ax:ax + image.shape[1]] = image
    return result


def RMSE(input, output):
    result = []
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            result.append((input[i][j] - output[i][j])**2)
    avg = sum(result)/len(result)
    return math.sqrt(avg)


def main():
    st.sidebar.title("""Tomograf""")

    patient_name = str(st.sidebar.text_input("Imię i Nazwisko", "Imie i Nazwisko"))
    patient_sex = str(st.sidebar.text_input("Płeć", "Plec"))
    patient_birth = str(st.sidebar.text_input("Data urodzenia", "01.01.2000"))
    patient_description = str(st.sidebar.text_input("Komentarz", "Brak"))

    czy_kroki_posrednie = st.sidebar.checkbox("Kroki pośrednie", value=True)
    filtrowanie = st.sidebar.checkbox("Filtr", value=True)

    input_file = st.sidebar.file_uploader("Upuść plik", type=['dcm', 'png', 'jpeg', 'jpg', 'bmp'])

    if czy_kroki_posrednie:
        krok = st.sidebar.slider("Postępu obrotu układu emitor/detektor", 1, 360, 1, 1)

    alfa = st.sidebar.slider("Krok α dla układu emiter/detektor", 0.5, 4.0, 1.0, 0.1)
    n = st.sidebar.slider("Liczba detektorów", 50, 720, 100, 10)
    l = st.sidebar.slider("Rozpiętość układu emiter/detektor", 10, 350, 270, 5)



    input_image = st.empty()
    sinogram_image = st.empty()
    output_image = st.empty()
    RMSE_text = st.empty()

    if (input_file != None):
        input_file_name = input_file.name

        if (input_file_name.split(".")[1] == "dcm"):
            ds = pydicom.dcmread(input_file_name);
            imgDcm = ds.pixel_array
            table = PrettyTable()
            attributes = [a for a in dir(ds) if "patient" in a.lower()[0:8] or "StudyDate" == a or "StudyDescription" == a]
            for attr in attributes:
                name_value = str(getattr(ds, attr)).replace("^", " ")
                attr2 = " ".join(re.findall('[A-Z][^A-Z]*', attr))
                table.add_row([attr2 + ": ", name_value])

            table.header = False
            table.border = False
            table.min_width["Field 1"] = 20
            table.min_width["Field 2"] = 20
            table.align["Field 1"] = "l"
            table.align["Field 2"] = "l"

            st.text(table)
            input_image.image(imgDcm, width=500, caption="")

        else:
            input_image.image(input_file, width=500, caption='Input')

            bitmap = np.asarray(bytearray(input_file.read()), dtype=np.uint8) # bitmapa
            image = cv2.imdecode(bitmap, 1)
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = square(image)

            image = normalize(image)

            sinogram = []
            output = np.zeros(image.shape)
            liczba_skanów = int(360 / alfa)

            emiter = []

            for i in np.linspace(0, 359, liczba_skanów):

                # obracanie emitera i detektorów, wyliczanie wspolrzednych emitera (podejscie stozkowe)
                r = int(image.shape[0]/2) - 1 #połowa szerokości

                emiter.clear()
                emiter = [r * np.cos(math.radians(i)), r * np.sin(math.radians(i))] #x,y
                detector_list = get_detector_positions(r, i, l, n)

                tmp_output = np.zeros(image.shape)
                sinogram_row = []
                list_of_points = [] #wszystkie punkty z emitora

                for detector in detector_list:
                    pixel, points = avg_pixel(emiter, detector, image)
                    sinogram_row.append(pixel)
                    list_of_points.append(points)

                sinogram.append(sinogram_row)

                if filtrowanie:
                    sinogram_row = filter(sinogram_row)

                for i in range(len(list_of_points)):
                        for [x, y] in list_of_points[i]:
                            tmp_output[x][y] += sinogram_row[i]

                output += tmp_output #dodanie obrazu z 1 emitera do wyniku

                if czy_kroki_posrednie:
                    if i % krok < 1 and i != 0:
                        sinogram_image.image(normalize(sinogram), caption='Sinogram')
                        output_image.image(normalize(output), width=500, caption='Output')


            sinogram_image.image(normalize(sinogram), caption='Sinogram')
            output = normalize(output)
            output = skimage.filters.rank.median(output, np.ones([3, 3]))
            output_image.image(output, width=500, caption='Output')

            RMSE_text.text(str(RMSE(image, output)).replace(".", ","))
            dicom(output, patient_name, patient_sex, patient_birth, patient_description)


if __name__ == "__main__":
    main()
