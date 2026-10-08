[1mdiff --git a/src/mini_mlops/data.py b/src/mini_mlops/data.py[m
[1mindex 2368217..038144e 100644[m
[1m--- a/src/mini_mlops/data.py[m
[1m+++ b/src/mini_mlops/data.py[m
[36m@@ -3,6 +3,8 @@[m [mfrom pathlib import Path[m
 import pandas as pd[m
 from sklearn.model_selection import train_test_split[m
 [m
[32m+[m[32mfrom mini_mlops.contracts import validate_training_data[m
[32m+[m
 REQUIRED_COLUMNS = {"text", "label"}[m
 [m
 def load_data(path : Path) -> pd.DataFrame:[m
[36m@@ -12,6 +14,8 @@[m [mdef load_data(path : Path) -> pd.DataFrame:[m
 [m
 def validate_schema(df: pd.DataFrame) -> None:[m
     """Perform basic schema validation."""[m
[32m+[m[32m    validate_training_data(df)[m
[32m+[m
     missing_columns = REQUIRED_COLUMNS - set(df.columns)[m
 [m
     if missing_columns:[m
[1mdiff --git a/src/mini_mlops/lineage.py b/src/mini_mlops/lineage.py[m
[1mindex edbbc56..c89e9a1 100644[m
[1m--- a/src/mini_mlops/lineage.py[m
[1m+++ b/src/mini_mlops/lineage.py[m
[36m@@ -46,4 +46,5 @@[m [mdef sha256_file(path: Path) -> str:[m
         while chunk := file.read(8192):[m
             sha256.update(chunk)[m
 [m
[32m+[m[32m    #  return "0" * 64[m
     return sha256.hexdigest()[m
\ No newline at end of file[m
