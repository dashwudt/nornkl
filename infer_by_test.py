import os
import sys
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# model_path - путь к вашей модели (с именем и расширением файла, относительно скрипта в вашем архиве проекта)
# dataset_path - путь к папке с тестовым датасетом.
# Он состоит из n фотографий c расширением .jpg (гарантируется, что будет только это расширение)
#
# output_path - путь к файлу, в который будут сохраняться результаты (с именем и расширением файла)
dataset_path, output_path = sys.argv[1:]

v_ignore = 5
# Число кусков по стоке и столбцу всего будет в квадрете штук
v_count_parts = 3
# Пороговый контркт
v_contrast_border = 369
#Режу картинку на куски [p_part_size Х p_part_size]
def f_make_massive_of_parts(p_img, p_part_size):
  v_res = []
  v_rows_size = round(p_img.shape[0] / p_part_size)
  v_col_size = round(p_img.shape[1] / p_part_size)
  for j in range(0, p_img.shape[0], v_rows_size):
    for i in range(0, p_img.shape[1], v_col_size):
      v_res.append(np.array(p_img[j:j + v_rows_size, i:i + v_col_size, 0:3], int))
  return v_res

# Определяем контраст ОДНОГО куска картинки
def f_make_contrast_of_part(p_mass, p_ignore):
    v_total_colors = (p_mass[:, :, 0] + p_mass[:, :, 1] + p_mass[:, :, 2]).tolist()
    v_res = []
    for i_str in v_total_colors:
      v_res += i_str
    v_res = sorted(v_res)
    v_ignore = round(len(v_res) * (p_ignore / 100))
    v_res = v_res[v_ignore:(-1 * v_ignore)]
    return max(v_res) - min(v_res)

def is_it_clear(v_img):
  v_parted_img = f_make_massive_of_parts(v_img,v_count_parts)
  # Формируем лист контрастностей кусков
  v_contrast_mass = []
  for i in v_parted_img:
    v_contrast_mass.append(f_make_contrast_of_part(i,v_ignore))
  #Определяем процент кусков с низким контрастом
  v_count_less = 0
  for i in v_contrast_mass:
      if i <= v_contrast_border:
        v_count_less+=1
  v_count_less /=(v_count_parts**2)
  if v_count_less<.75:
    it_is_clear=True
  else:
    it_is_clear=False
  return it_is_clear











# TODO ваша работа с моделью
# на вход модели подаются изображения из тестовой выборки
# результатом должен стать json-файл
# В качестве примера здесь показана работа на примере модели из baseline

# Пример функции инференса модели
def infer_image(model, image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Инференс
    return model(image)


# TODO Ваш проект будет перенесен целиком, укажите корректны относительный путь до модели!!!
# TODO Помните, что доступа к интернету не будет и нельзя будет скачать веса модели откуда-то с внешнего ресурса!
model_path = './best.pt'

# Тут показан пример с использованием модели, полученной из бейзлайна
example_model = YOLO(model_path)
example_model.to('cpu')


def create_mask(image_path, results, flag):
    # Загружаем изображение и переводим в градации серого
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Создаем пустую маску с черным фоном
    mask = np.zeros((height, width), dtype=np.uint8)

    if flag:
        return mask

    # Проходим по результатам и создаем маску
    for result in results:
        masks = result.masks  # Получаем маски из результатов
        if masks is not None:
            for mask_array in masks.data:  # Получаем маски как массивы
                mask_i = mask_array.numpy()  # Преобразуем маску в numpy массив

                # Изменяем размер маски под размер оригинального изображения
                mask_i_resized = cv2.resize(mask_i, (width, height), interpolation=cv2.INTER_LINEAR)

                # Накладываем маску на пустую маску (255 для белого)
                mask[mask_i_resized > 0] = 255

    return mask

# Ваша задача - произвести инференс и сохранить маски НЕ в отдельные файлы, а в один файл submit.
# Для этого мы сначала будем накапливать результаты в словаре, а затем сохраним их в JSON.
results_dict = {}

for image_name in os.listdir(dataset_path):
    if image_name.lower().endswith(".jpg"):
        results = infer_image(example_model, os.path.join(dataset_path, image_name))
        it_is_clear = is_it_clear(cv2.imread(os.path.join(dataset_path, image_name)))
        mask = create_mask(os.path.join(dataset_path, image_name), results, it_is_clear)
        
        # Кодируем маску в PNG в память
        _, encoded_img = cv2.imencode(".png", mask)
        # Кодируем в base64, чтобы поместить в JSON
        encoded_str = base64.b64encode(encoded_img).decode('utf-8')
        results_dict[image_name] = encoded_str

# Сохраняем результаты в один файл "submit" (формат JSON)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False)
