from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from utils.createTSNE import create_tsne_plot
import numpy as np
import os
import pandas as pd


def extract_root_id(mol_id: str) -> str:
    # Split on the first hyphen and take the element before it, if any.
    # If there's no hyphen, it just returns the original ID.
    return mol_id.split('-')[0]


def compute_category_metrics(y_true, y_pred, y_proba, act_mol_values, act_conf_values):
    """
    Compute metrics for each category of instances:
    1) Active conformer of active molecule (act_mol=1, act_conf=1)
    2) Inactive conformer of active molecule (act_mol=1, act_conf=0)
    3) Inactive conformer of inactive molecule (act_mol=0, act_conf=0)

    Args:
        y_true (array-like): True activity labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities for the positive class (or None if not available).
        act_mol_values (array-like): The act_mol column values.
        act_conf_values (array-like): The act_conf column values.
    """

    def metrics_for_subset(subset_idx):
        y_true_sub = y_true[subset_idx]
        y_pred_sub = y_pred[subset_idx]
        y_proba_sub = y_proba[subset_idx] if y_proba is not None else None

        bal_acc = balanced_accuracy_score(y_true_sub, y_pred_sub)
        prec = precision_score(y_true_sub, y_pred_sub, zero_division=0)
        rec = recall_score(y_true_sub, y_pred_sub, zero_division=0)
        f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)
        roc = roc_auc_score(y_true_sub, y_proba_sub) if y_proba_sub is not None else "N/A"
        return {
            "Balanced Accuracy": bal_acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
        }

    # Define subsets based on conditions
    cat1_idx = (act_mol_values == 1) & (act_conf_values == 1)  # active conf of active mol
    cat2_idx = (act_mol_values == 1) & (act_conf_values == 0)  # inactive conf of active mol
    cat3_idx = (act_mol_values == 0) & (act_conf_values == 0)  # inactive conf of inactive mol

    results = {}
    if np.any(cat1_idx):
        results["ActiveConf_ActiveMol"] = metrics_for_subset(cat1_idx)
    if np.any(cat2_idx):
        results["InactiveConf_ActiveMol"] = metrics_for_subset(cat2_idx)
    if np.any(cat3_idx):
        results["InactiveConf_InactiveMol"] = metrics_for_subset(cat3_idx)

    return results


def calculate_confusion_matrix(subset):
    """
    Calculate the confusion matrix for a given subset.
    Args:
        subset (pd.DataFrame): A subset of the dataset.

    Returns:
        dict: Confusion matrix values.
    """
    tp = ((subset['act_conf'] == 1) & (subset['prediction'] == 1)).sum()
    fn = ((subset['act_conf'] == 1) & (subset['prediction'] == 0)).sum()
    fp = ((subset['act_conf'] == 0) & (subset['prediction'] == 1)).sum()
    tn = ((subset['act_conf'] == 0) & (subset['prediction'] == 0)).sum()

    return {
        'TP': tp,
        'FN': fn,
        'FP': fp,
        'TN': tn
    }


def calculate_f1(cm):
    """
    Calculate F1 Score for positive instances.
    Args:
        cm (dict): Confusion matrix values.

    Returns:
        float: F1 Score.
    """
    precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0
    recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def calculate_balanced_accuracy(cm):
    """
    Calculate Balanced Accuracy for negative instances of positive bags.
    Args:
        cm (dict): Confusion matrix values.

    Returns:
        float: Balanced Accuracy.
    """
    specificity = cm['TN'] / (cm['TN'] + cm['FP']) if (cm['TN'] + cm['FP']) > 0 else 0
    sensitivity = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
    return (specificity + sensitivity) / 2


def calculate_weighted_nip_negative(cm):
    """
    Calculate Weighted Negative Instance Precision (Weighted NIP) for negative instances of negative bags.
    Args:
        cm (dict): Confusion matrix values.

    Returns:
        float: Weighted NIP value.
    """
    tn = cm['TN']
    fp = cm['FP']
    return tn / (tn + 2 * fp) if (tn + 2 * fp) > 0 else 0


def calculate_composite_score(f1, balanced_acc, weighted_nip_negative):
    """
    Calculate a composite score combining the three metrics.
    Args:
        f1 (float): F1 score.
        balanced_acc (float): Balanced accuracy.
        weighted_nip_negative (float): Weighted NIP for negative instances of negative bags.

    Returns:
        float: Composite score.
    """
    # Adjusted weights to prioritize F1 Score more
    # return 0.5 * balanced_acc + 0.3 * f1 + 0.2 * weighted_nip_negative
    return balanced_acc + f1 + weighted_nip_negative


class QSARModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        print("Initializing QSARModel...")
        self.X_train_full = X_train
        self.y_train_df_full = y_train

        self.X_test = X_test
        self.y_test_df = y_test
        self.y_test = y_test['act_conf'].values
        self.test_mol_ids = y_test['mol_id']

        print("Initialization complete.")

    def run_models(self, use_test_set=False, fp_name=None, output_dir=None, use_custom_scorer=True):
        print("Splitting data into train and validation sets...")

        # Ensure active/decoy pairs are in the same fold
        groups = self.y_train_df_full['mol_id'].apply(extract_root_id)

        # Use GroupShuffleSplit for train-validation split
        group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, val_indices = next(group_splitter.split(self.X_train_full, self.y_train_df_full['act_conf'], groups))

        X_train = self.X_train_full[train_indices]
        y_train_df = self.y_train_df_full.iloc[train_indices]
        y_train = y_train_df['act_conf'].values
        train_mol_ids = y_train_df['mol_id']

        X_val = self.X_train_full[val_indices]
        y_val_df = self.y_train_df_full.iloc[val_indices]
        y_val = y_val_df['act_conf'].values

        print("Data splitting complete.")

        print("Setting up models...")
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=1000, n_jobs=-1, class_weight="balanced"
            )
        }

        param_grids = {
            "Random Forest": {
                "max_depth": [10, 30, 50]
            }
        }

        group_kfold = GroupKFold(n_splits=5)

        results = {}

        for model_name, model in models.items():
            print(f"Running group-based cross-validation for {model_name}...")

            scorer = self.custom_scorer if use_custom_scorer else 'balanced_accuracy'

            grid_search = GridSearchCV(
                model,
                param_grid=param_grids.get(model_name, {}),
                cv=group_kfold.split(X_train, y_train, groups=train_mol_ids),
                scoring=scorer,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            print(f"Best Parameters for {model_name}: {grid_search.best_params_}")

            # Fit metrics on training set
            print(f"Calculating fit metrics for {model_name} on the training set...")
            y_train_pred = best_model.predict(X_train)
            y_train_proba = (best_model.predict_proba(X_train)[:, 1]
                             if hasattr(best_model, "predict_proba") else None)

            train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            train_roc_auc = (roc_auc_score(y_train, y_train_proba)
                              if y_train_proba is not None else "N/A")

            print(f"Training Set Results for {model_name}:")
            print(f"  Balanced Accuracy: {train_balanced_acc:.4f}")
            print(f"  Precision: {train_precision:.4f}")
            print(f"  Recall: {train_recall:.4f}")
            print(f"  F1-Score: {train_f1:.4f}")
            print(f"  ROC-AUC: {train_roc_auc:.4f}" if y_train_proba is not None else "  ROC-AUC: N/A")
            print("-" * 40)

            # Evaluate on the validation set
            print(f"Evaluating {model_name} on the validation set...")
            y_val_pred = best_model.predict(X_val)
            y_val_proba = (best_model.predict_proba(X_val)[:, 1]
                           if hasattr(best_model, "predict_proba") else None)

            # Create CSV with results
            print("Saving validation predictions to CSV...")
            val_results_df = y_val_df.copy()
            val_results_df['prediction'] = y_val_pred
            val_results_df['probability'] = y_val_proba if y_val_proba is not None else None
            val_results_df[['mol_id', 'full_id', 'energy', 'act_mol', 'act_conf', 'prediction', 'probability']].to_csv(os.path.join(output_dir,
                f"Report_{fp_name}_custom2_validation_results_{model_name.replace(' ', '_').lower()}.csv"), index=False
            )
            print(f"Validation predictions saved to validation_results_{model_name.replace(' ', '_').lower()}.csv")

            val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, zero_division=0)
            val_recall = recall_score(y_val, y_val_pred, zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            val_roc_auc = (roc_auc_score(y_val, y_val_proba)
                            if y_val_proba is not None else "N/A")

            results[model_name] = {
                "Best Params": grid_search.best_params_,
                "Train Balanced Accuracy": train_balanced_acc,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Train F1-Score": train_f1,
                "Train ROC-AUC": train_roc_auc,
                "Validation Balanced Accuracy": val_balanced_acc,
                "Validation Precision": val_precision,
                "Validation Recall": val_recall,
                "Validation F1-Score": val_f1,
                "Validation ROC-AUC": val_roc_auc,
            }

            print(f"Validation Set Results for {model_name}:")
            print(f"  Balanced Accuracy: {val_balanced_acc:.4f}")
            print(f"  Precision: {val_precision:.4f}")
            print(f"  Recall: {val_recall:.4f}")
            print(f"  F1-Score: {val_f1:.4f}")
            print(f"  ROC-AUC: {val_roc_auc:.4f}" if y_val_proba is not None else "  ROC-AUC: N/A")
            print("-" * 40)

            if use_test_set:
                print(f"Evaluating {model_name} on the test set...")
                y_test_pred = best_model.predict(self.X_test)
                y_test_proba = (best_model.predict_proba(self.X_test)[:, 1]
                                if hasattr(best_model, "predict_proba") else None)

                test_balanced_acc = balanced_accuracy_score(self.y_test, y_test_pred)
                test_precision = precision_score(self.y_test, y_test_pred, zero_division=0)
                test_recall = recall_score(self.y_test, y_test_pred, zero_division=0)
                test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
                test_roc_auc = (roc_auc_score(self.y_test, y_test_proba)
                                if y_test_proba is not None else "N/A")

                results[model_name].update({
                    "Test Balanced Accuracy": test_balanced_acc,
                    "Test Precision": test_precision,
                    "Test Recall": test_recall,
                    "Test F1-Score": test_f1,
                    "Test ROC-AUC": test_roc_auc,
                })

                print(f"Test Set Results for {model_name}:")
                print(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
                print(f"  Precision: {test_precision:.4f}")
                print(f"  Recall: {test_recall:.4f}")
                print(f"  F1-Score: {test_f1:.4f}")
                print(f"  ROC-AUC: {test_roc_auc:.4f}" if y_test_proba is not None else "  ROC-AUC: N/A")
                print("-" * 40)

        return results

    def custom_scorer(self, estimator, X, y):
        predictions = estimator.predict(X)
        probabilities = estimator.predict_proba(X)[:, 1] if hasattr(estimator, 'predict_proba') else None
        df = pd.DataFrame({
            'act_conf': y,
            'prediction': predictions
        })

        cm = calculate_confusion_matrix(df)
        f1 = calculate_f1(cm)
        balanced_acc = calculate_balanced_accuracy(cm)
        weighted_nip_negative = calculate_weighted_nip_negative(cm)

        composite_score = calculate_composite_score(f1, balanced_acc, weighted_nip_negative)

        return composite_score
