# README

# Metrics
MSE for regression
AUC for classification

|dataset|baseline metric|R^2| catboost |
|---|---|---|
|`check_1_r`|131.07971290487552|0.6579522614182832| 0.64 (overfitted) |
|`check_2_r`|2.6693092445594617|0.4953323080395863| 0.64 |
|`check_3_r`|13986292721.637714|0.656515919949344| 0.62 |
|`check_4_с`|0.861895210575268|-| failed |
|`check_5_c`|0.7717664972379903|-| 0.784 |
|`check_6_c`|0.6551207582009038|-| 0.659 |
|`check_7_c`|0.7269461921229332|-| failed (le) |
|`check_8_c`|0.862747736755639|-| failed (le)|

# About features
|dataset|date|datetime|number-float|number-int|string|id|columns with na|
|---|---|---|---|---|---|
|`check_1_r`|1|0|38|0 ?|0|0|0|
|`check_2_r`|0|0|4|1|2 (city, weekday)| 0 | 2|
|`check_3_r`|1|0|1|37|0|0|8|
|`check_4_с`|3|0|137|0 ?|0|0|3-4|
|`check_5_c`|1|0|9|4 |0|0|0|
|`check_6_c`|0|0|111|0 ?|0|0|3-7|
|`check_7_c`|2|0|764| ? |4|1|куча|
|`check_8_c`|90|?|752| ? |29|0|куча|
