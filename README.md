# itmo_image_rec_gen
ITMO Image recognition &amp; generation course
___
## Лабораторная работа № 1 «Классификация изображений с помощью сверточных нейронных сетей»
Набор данных Гербарий 2022: Флора Северной Америки является частью проекта Ботанического сада Нью-Йорка, финансируемого Национальным научным фондом, по созданию инструментов для выявления новых видов растений по всему миру. Набор данных стремится представить все известные таксоны сосудистых растений в Северной Америке, используя изображения, собранные из 60 различных ботанических учреждений по всему миру.
Набор данных Herbarium 2022: Flora of North America https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data содержит 1,05 млн изображений 15 500 сосудистых растений, которые составляют более 90% таксонов, задокументированных в Северной Америке. Использовался контрольный список сосудистых растений Америки (VPA), подготовленный Ботаническим садом Миссури. Набор данных ограничен тем, что включает только сосудистые наземные растения (плауновидные, папоротники, голосеменные и цветковые растения).
Необходимо построить классификатор изображений сосудистых растений. Оценка качества производится при помощи F1_macro, притом F1_macro > 0.8.

Если у вас сложности со скачиванием набора данных, то можно использовать Fruits 360 https://www.kaggle.com/datasets/moltean/fruits 

Уточнение: только классификатора на основе ResNet с переносом знаний недостаточно.
___
## Лабораторная работа № 2 «Детектирование объектов на изображениях с помощью глубоких нейронных сетей»
В практически любой точке современного города человека окружает множество номеров и надписей. Обучите (или дообучите) нейронную сеть детектировать номера на изображениях (номер представляет собой последовательность длиной до 20 символов из цифр, прописных и строчных букв русского или английского алфавита, знаков дефис и слеш).  Сфотографируйте номера на улицах города (не из Интернета, проверим поиском по картинке) и запустите на них свой алгоритм. Оценка качества детекции производится с помощью метрик IoU, Precision, Recall, mAP на тестовой части набора The Street View House Numbers http://ufldl.stanford.edu/housenumbers/.
___
## Лабораторная работа № 3 «Сегментация изображений с помощью глубоких нейронных сетей»
Предобученные модели детекции, такие как Mask R-CNN детектируют и сегментируют объекты фиксированных классов, в том числе знаки «Стоп», в этой лабораторной предлагается обучить выбранную модель сегментировать 8 типов знаков на наборе данных Russian road signs https://www.kaggle.com/datasets/viacheslavshalamov/russian-road-signs-segmentation-dataset, содержащем 100000 изображений дорожных знаков России (набор для сегментации построен на основе набора данных для детекции https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html ). Предобученные модели можно использовать из библиотеки MMDetection https://github.com/open-mmlab/mmdetection.   
Тестирование провести на 10 фотографиях улиц с дорожными знаками, сделанных самостоятельно около Университета ИТМО, если невозможно по уважительным причинам, то в других узнаваемых местах – эти фотографии с предсказанной сегментацией нужно будет показать проверяющему.  Оценка качества сегментации производится с помощью метрик IoU, Precision, Recall, L2 отдельно на валидационной части набора изображений Russian road signs и сделанных фотографиях, также интересует процент изображений, на котором IoU >= 0.5, IoU >= 0.75, IoU >= 0.9.
___
## Лабораторная работа № 4 «Генерация изображений по сегментации с помощью глубоких нейронных сетей»
С помощью дообученной самостоятельно модели SPADE https://github.com/NVlabs/SPADE вам предлагается сгенерировать несколько изображений (3-5 шт) на основе сегментации фотографии вашего жилища, вида из окна или любого предмета в вашем окружении (при проверке нужно будет показать это в веб-камеру).
Можно воспользоваться готовой моделью для сегментации, например https://github.com/CSAILVision/semantic-segmentation-pytorch или https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py 
___
## Лабораторная работа № 5 «Генерация изображений по тексту с помощью глубоких нейронных сетей»
С помощью предобученной модели DALL-E (например https://github.com/ai-forever/ru-dalle, не по API!) вам предлагается сгенерировать изображение (размером не менее 512х512 пикселей) по тексту широко известного художественного произведения (например, по отрывку из Войны и мира, Ведьмака или Игры престолов). Сначала используйте цитату в два предложения, потом постепенно уменьшайте и уточнейте запрос, чтобы получить более точную иллюстрация (показать при проверке минимум 5 шагов).
