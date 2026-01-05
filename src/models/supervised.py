import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from typing import List
import matplotlib.pyplot as plt 
from src.utils.mlflow_client import log_metric


class SupervisedModeler:
    """
    Gerencia o treinamento e avaliação de modelos supervisionados.
    Padrão: Strategy Pattern (pode receber qualquer estimador sklearn).
    """
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        self.models = {}

    def train_evaluate(self, model_name: str, model_instance):
        """Treina e avalia um modelo específico."""
        print(f"\n🚀 Treinando {model_name}...")
        model_instance.fit(self.X_train, self.y_train)
        preds = model_instance.predict(self.X_test)
        
        # Calcula métricas
        acc = accuracy_score(self.y_test, preds)
        prec = precision_score(self.y_test, preds, zero_division=0)
        rec = recall_score(self.y_test, preds, zero_division=0)
        f1 = f1_score(self.y_test, preds, zero_division=0)

        # Loga no MLflow (se houver run ativa)
        log_metric("accuracy", acc)
        log_metric("precision", prec)
        log_metric("recall", rec)
        log_metric("f1", f1)

        print(f"📊 Relatório para {model_name}:")
        print(classification_report(self.y_test, preds))
        
        self.models[model_name] = model_instance
        return model_instance

    def plot_feature_importance(self, model_name: str, feature_names: List[str]):
        """Plota importância das features (apenas para modelos baseados em árvore)."""
        model = self.models.get(model_name)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 5))
            plt.title(f"Feature Importance - {model_name}")
            plt.bar(range(len(indices)), importances[indices], align="center")
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
