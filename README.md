# [IT Purple Hack - Solving CLTV tasks](https://eval.ai/web/challenges/challenge-page/2228/overview)

Winner of the 3rd place - [link to the presentation](https://drive.google.com/drive/folders/1v8k7YrlSVRtAc2gshzRDfNkWN9KZ3YM-?usp=sharing) 

This solution allows predicting the probability of a bank client (legal entity) transitioning to each of the 17 product clusters over a 12-month horizon. We conducted in-depth research into the application of AI in CLTV tasks, which can contribute to the development of new solutions in this area.

# The final solution consists of:
- **CatBoost as a multiclass classifier** for cluster prediction. Data aggregation by client, One-hot and label-encoder techniques were used to improve metrics. The training dataset was transformed, and features were selected. At this stage, the model showed a score of approximately 0.87 on the public dataset. After a series of experiments with the model, we tried changing the encoding of categorical features, which improved the model score to **0.89752**.
- **Cascade of 22 binary models**. The method is based on training models to predict the churn probability and propensity of individual products. The final score of this model was **0.86086**. Adding the company churn model as a feature, new features, replacing CatBoost with LightGBM, and other experiments with the model did not improve the score.
- Ensemble of three models: **linear regression, LightGBM, CatBoost**. The public score of the resulting ensemble is **0.87257**.

We continued experimenting and tried to combine these models, which consistently led to score improvements. The ensemble of the Multiclass Classifier and Cascade brought us a score of 0.90001. This result was also combined with the latest ensemble, resulting in our final leaderboard score of **0.90155**.

We also created a model **to predict start_cluster in month_6** of the test dataset. [The resulting dataset](https://drive.google.com/file/d/1IduKs5XyuIBH9LH-WzFBFrRBktYQXSju/view?usp=sharing), applied to the above models, slightly improved their results by fractions of thousandths or hundredths. It was also used in our final solution.

# Description of main files
- baseline_valya_binary_kaskad.ipynb *Cascade of binary models*
- baseline_alina_score90.ipynb *Multiclass classifier*
- baseline_newv2.ipynb *Ensemble*
- coeff_model_sub.py *Output*
  
Other files represent additional research conducted by us within the hackathon.

# Additional research
- **MultinomialHMM model** showed the lowest score - 0.61.
- **NaiveNB** and **KNN models** were used to create additional features but did not fit well and showed weak results (from 0.7 to 0.8).
- For the **RandomForestClassifier model**, parameters were tuned using Optuna but yielded a score no higher than 0.85 - 0.86. Polynomial features were tried, which reduced the result to 0.8. Experiments with **ImbalancedRandomForestClassifier** and segments also reduced the result.
- The **TabNet neural network** yielded a result of about 0.78. We also wanted to experiment with other neural networks like RNN, but we didn't have time to do so within the hackathon. However, we found it interesting to present the task as a well-known next-basket prediction problem and explore similar methods.
- Ensemble of **LightGBM + LightGBM with Optuna**, score 0.87666.
  
# Team [MISIShunters](https://misishunters.website.yandexcloud.net)
- [Alisa Semenova](https://t.me/NeAlyssa)
  Project Manager, Analyst || 4th year, Institute of Computer Science, NUST MISIS
  
- [Alina Burykina](https://t.me/BurykinaA)
  ML Engineer, Backend developer || 4th year, Institute of Computer Science, NUST MISIS
  
- [Valentina Nikolaeva](https://t.me/qswder)
  Data Scientist, OZON Fintech || Graduate, RANEPA, Economics and Finance
  
- [Ekaterina Mezhuieva](https://t.me/tg_katyaa)
  Data Scientist, Department of Information Technologies, Moscow City || Graduate, RANEPA, Economics and Finance
  
- [Elizaveta Borisenko](https://t.me/kokosikEH)
  Designer, Frontend Developer || 4th year, Institute of Computer Science, NUST MISIS
