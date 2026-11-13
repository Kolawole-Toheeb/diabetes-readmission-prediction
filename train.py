import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/diabetic_data.csv'
OUTPUT_FILE = 'model.bin'
TARGET = 'readmit_30'

XGB_PARAMS = {
    'eta': 0.1,
    'max_depth': 5,
    'min_child_weight': 5,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 0,
}
NUM_ROUNDS = 200


def diag_group(code):
    """Group an ICD-9 diagnosis code into a broad disease group."""
    if code == 'Unknown':
        return 'Unknown'
    if str(code).startswith(('V', 'E')):
        return 'Other'
    try:
        n = float(code)
    except ValueError:
        return 'Other'
    if 390 <= n <= 459 or n == 785: return 'Circulatory'
    if 460 <= n <= 519 or n == 786: return 'Respiratory'
    if 520 <= n <= 579 or n == 787: return 'Digestive'
    if 250 <= n < 251: return 'Diabetes'
    if 800 <= n <= 999: return 'Injury'
    if 710 <= n <= 739: return 'Musculoskeletal'
    if 580 <= n <= 629 or n == 788: return 'Genitourinary'
    if 140 <= n <= 239: return 'Neoplasms'
    return 'Other'


def prepare_data(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.replace('?', pd.NA)

    # target: readmitted within 30 days
    df[TARGET] = (df['readmitted'] == '<30').astype(int)

    # remove patients who died or went to hospice
    df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

    # keep one visit per patient
    df = df.sort_values('encounter_id').drop_duplicates(subset='patient_nbr', keep='first')

    df = df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'readmitted'])

    for c in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        df[c] = df[c].astype(str)

    for c in df.select_dtypes(exclude='number').columns:
        df[c] = df[c].fillna('Unknown')

    for c in ['diag_1', 'diag_2', 'diag_3']:
        df[c] = df[c].apply(diag_group)

    return df


def main():
    print('Preparing the data...')
    df = prepare_data(DATA_PATH)
    print(f'Rows after cleaning: {len(df)}')

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df[TARGET]
    )

    features = [c for c in df.columns if c != TARGET]

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(df_full_train[features].to_dict(orient='records'))
    X_test = dv.transform(df_test[features].to_dict(orient='records'))

    print('Training the final XGBoost model...')
    dtrain = xgb.DMatrix(X_full_train, label=df_full_train[TARGET].values)
    dtest = xgb.DMatrix(X_test)
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)

    auc = roc_auc_score(df_test[TARGET].values, model.predict(dtest))
    print(f'Test AUC: {auc:.4f}')

    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f'Model saved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()