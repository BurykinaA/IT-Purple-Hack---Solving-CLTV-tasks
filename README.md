# IT Purple Hack - Solving CLTV tasks

Это решение, позволяющее предсказывать вероятность перехода клиента банка (юр. лица) в каждый из 17 продуктовых кластеров на горизонте 12 месяцев. Мы провели глубокое исследование возможностей применения ИИ в задачах с CLTV, которое может внести вклад в развитие новых решений в этой области.

# Итоговое решение состоит из:
- **CatBoost в качестве мультиклассификатора** для предсказания кластера. Для повышение метрики использовались one-hot и label-encoder. Был преобразован train датасет и отобраны признаки. На данном этапе на публичном датасете модель показывала скор ~0.87. После ряда экспериментов с моделью мы попробовали изменить кодировку категориальных признаков, что улучшило скор модели до **0.89752**.
- **Каскад бинарных моделей**. Метод построен на обучении моделей, предсказывающих вероятность оттока и склонности отдельных продуктов. Итоговый скор этой модели составил **0.86086**. Добавление модели оттока компании как признака, новые признаки, замена catboost на lightjbm и другие эксперименты с моделью в результате скор не улучшили.
- Ансамбль трех моделей: **линейная регрессия, lightgbm, catboost**. Публичный скор получившегося ансамбля - **0.87257**. Дальше использовали optuna, (добавим рез когда обучится).

Далее мы продолжили эксперименты и пробовали объединить эти модели между собой, что стабильно приводило к повышению скора. Ансамбль из Мультиклассификатора и Каскада принес нам скор 0.90001. Этот результат мы также объединили с последним ансамблем, в результате чего добились нашего итогового скора на лидерборде - **0.90155**.

Также мы сделали модель **для предсказания start_cluster в month_6** тестового датасета. [Получившийся датасет](https://drive.google.com/file/d/1IduKs5XyuIBH9LH-WzFBFrRBktYQXSju/view?usp=sharing), примененный к каждой модели выше, незначительно повышал их результат на тысячные-сотые доли. Использовалась также в нашем итоговом решении.

# Описание основных файлов
- baseline_valya_binary_kaskad.ipynb *Каскад бинарных моделей*
- baseline_alina_score90.ipynb *Мультиклассификатор*
- baseline_newv2.ipynb *Ансамбль*

Остальные файлы представляют дополнительные исследования, проведенные нами в рамках хакатона.

# Дополнительные исследования
- Модель **MultinomialHMM** показала самый низкий скор - 0.61. 
- Модели **NaiveNB** и **KNN** использовались для создания дополнительных признаков, но не подошли и показали слабый результат (от 0.7 до 0.8). 
- Для модели **RandomForestClassifier** параметры подбирались с помощью optuna, но дал скор не выше 0.85 - 0.86. Пробовали полиномиальные признаки – снижало результат до 0.8. Эксперименты с **ImbalancedRandomForestClassifier** и сегментами также снижали результат. 
- Нейросеть **TabNet** дала результат около 0.7. Мы хотели также поэкспериментировать с другими нейросетями, как RNN, но не успели заняться этим в рамках хакатона. Но нам показалось интересным представить задачу кейса, как известную задачу next-basket prediction, и исследовать схожие методы.

# Команда MISIShunters
- [Алиса Семенова](https://t.me/NeAlyssa)
  Project Manager, Analyst || 4 курс Института компьютерных наук НИТУ МИСИС
  
- [Алина Бурыкина](https://t.me/BurykinaA)
  ML, Backend developer || 4 курс Института компьютерных наук НИТУ МИСИС
  
- [Валентина Николаева](https://t.me/qswder)
  Data Scientist, OZON Fintech || Выпускница РАНХиГС, экономика и финансы 
  
- [Екатерина Межуева](https://t.me/tg_katyaa)
  Data Scientist, Департамент Информационных технологий г. Москвы || Выпускница РАНХиГС, экономика и финансы
  
- [Елизавета Борисенко](https://t.me/kokosikEH)
  Designer, Frontend developer || 4 курс Института компьютерных наук НИТУ МИСИС 
