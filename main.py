import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib
import matplotlib.pyplot as plt
import time
import os, glob
from datetime import datetime
import threading
import logging

matplotlib.use('Agg')

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction pour générer de nouvelles données avec des déviations occasionnelles
def generate_data(n_samples, deviation_probability=0.1, deviation_factor=3):
    # Générer des données normales
    data = {
        'cpu': np.random.normal(50, 10, n_samples),
        'server_load': np.random.normal(50, 10, n_samples),
        'ram': np.random.normal(50, 10, n_samples)
    }

    # Injecter des déviations
    for key in data:
        # Générer un masque aléatoire pour les déviations
        deviation_mask = np.random.rand(n_samples) < deviation_probability
        # Appliquer les déviations
        data[key][deviation_mask] += np.random.normal(0, deviation_factor * 10, deviation_mask.sum())

    return data

# Créer les répertoires s'ils n'existent pas
os.makedirs('30SecPlotting', exist_ok=True)

# Initialiser le DataFrame avec les données initiales
data = generate_data(60)
df = pd.DataFrame(data)

# Fonction pour détecter les anomalies et tracer les graphiques
def detect_and_plot(df, file_name):
    logger.info(f"Détection et traçage des anomalies, enregistrement dans : {file_name}")
    df = df.copy()

    # Initialiser les modèles Isolation Forest pour chaque composant
    iso_forest_cpu = IsolationForest(contamination=0.25, random_state=42)
    iso_forest_server_load = IsolationForest(contamination=0.25, random_state=42)
    iso_forest_ram = IsolationForest(contamination=0.25, random_state=42)

    # Entraîner les modèles Isolation Forest
    df['cpu_anomaly'] = iso_forest_cpu.fit_predict(df[['cpu']])
    df['server_load_anomaly'] = iso_forest_server_load.fit_predict(df[['server_load']])
    df['ram_anomaly'] = iso_forest_ram.fit_predict(df[['ram']])

    # Convertir -1 (anomalie) en 1 et 1 (normal) en 0 pour un tracé plus facile
    df['cpu_anomaly'] = df['cpu_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    df['server_load_anomaly'] = df['server_load_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    df['ram_anomaly'] = df['ram_anomaly'].apply(lambda x: 1 if x == -1 else 0)

    # Combiner les anomalies pour l'état global
    df['combined_anomaly'] = df[['cpu_anomaly', 'server_load_anomaly', 'ram_anomaly']].max(axis=1)

    # Poids assignés à chaque composant
    weights = {
        'cpu_anomaly': 2,
        'server_load_anomaly': 3,
        'ram_anomaly': 1
    }

    # Calculer le score d'anomalie pondéré
    df['weighted_anomaly'] = (
        df['cpu_anomaly'] * weights['cpu_anomaly'] +
        df['server_load_anomaly'] * weights['server_load_anomaly'] +
        df['ram_anomaly'] * weights['ram_anomaly']
    ).astype(float)  # Cast explicite en float

    # Normaliser le score d'anomalie pondéré pour une meilleure visualisation
    max_weighted_score = sum(weights.values())
    df['weighted_anomaly'] = df['weighted_anomaly'] / max_weighted_score

    # Tracé des graphiques
    plt.figure(figsize=(15, 10))

    # Tracer les anomalies CPU
    plt.subplot(5, 1, 1)
    plt.scatter(df.index, df['cpu'], c=df['cpu_anomaly'], cmap='coolwarm', label='Anomalies CPU')
    plt.xlabel('Index')
    plt.ylabel('CPU')
    plt.legend()
    plt.title('Détection d\'anomalies CPU')

    # Tracer les anomalies de charge serveur
    plt.subplot(5, 1, 2)
    plt.scatter(df.index, df['server_load'], c=df['server_load_anomaly'], cmap='coolwarm', label='Anomalies Charge Serveur')
    plt.xlabel('Index')
    plt.ylabel('Charge Serveur')
    plt.legend()
    plt.title('Détection d\'anomalies Charge Serveur')

    # Tracer les anomalies RAM
    plt.subplot(5, 1, 3)
    plt.scatter(df.index, df['ram'], c=df['ram_anomaly'], cmap='coolwarm', label='Anomalies RAM')
    plt.xlabel('Index')
    plt.ylabel('RAM')
    plt.legend()
    plt.title('Détection d\'anomalies RAM')

    # Tracer les anomalies combinées
    plt.subplot(5, 1, 4)
    plt.plot(df.index, df['combined_anomaly'], label='Anomalies Combinées', color='red')
    plt.xlabel('Index')
    plt.ylabel('Anomalie')
    plt.legend()
    plt.title('Détection d\'anomalies Combinées')

    # Tracer les anomalies pondérées
    plt.subplot(5, 1, 5)
    plt.plot(df.index, df['weighted_anomaly'], label='Anomalies Pondérées', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Score d\'Anomalie Pondéré')
    plt.legend()
    plt.title('Détection d\'anomalies Pondérées')

    plt.tight_layout()

    # Enregistrer le graphique en PDF
    if file_name:
        plt.savefig(file_name, format='pdf')
    plt.close()

def monitoring_loop():
    global df
    while not stop_event.is_set():
        logger.info("Génération de nouvelles données et mise à jour du DataFrame.")
        try:
            time.sleep(30)
            new_data = generate_data(60)
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            logger.debug(f"Taille du DataFrame après concaténation : {df.shape}")
            
            last_30_seconds_df = df.tail(len(new_df))
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name_30s = f"30SecPlotting/last_30_seconds_{timestamp}.pdf"
            detect_and_plot(last_30_seconds_df, file_name_30s)
            logger.info(f"PDF des dernières 30 secondes généré : {file_name_30s}")
        except Exception as e:
            logger.error(f"Erreur pendant la surveillance: {str(e)}")

monitoring = False
stop_event = threading.Event()

# Démarrage et arrêt de la surveillance
while True:
    user_input = input("Entrez 'ON' pour démarrer la surveillance, 'OFF' pour arrêter la surveillance, 'PLOT_RANGE' pour générer les graphiques pour une plage d'index, 'EXIT' pour quitter : ").strip().upper()
    
    if user_input == 'ON' and not monitoring:
        monitoring = True
        logger.info("Surveillance démarrée.")
        stop_event.clear()
        monitoring_thread = threading.Thread(target=monitoring_loop)
        monitoring_thread.start()
    elif user_input == 'OFF' and monitoring:
        monitoring = False
        stop_event.set()
        monitoring_thread.join()
        logger.info("Surveillance arrêtée.")
    elif user_input == 'PLOT_RANGE':
        if not df.empty:
            try:
                max_index = len(df) - 1
                print(f"L'index maximum actuel est {max_index}.")
                start_idx = int(input("Entrez l'index de début : ").strip())
                end_idx = int(input("Entrez l'index de fin : ").strip())
                if start_idx < 0 or end_idx > max_index or start_idx > end_idx:
                    logger.warning("Indices invalides. Veuillez entrer des indices valides.")
                else:
                    plot_range_df = df.iloc[start_idx:end_idx+1]
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    file_name_range = f"30SecPlotting/range_data_{timestamp}.pdf"
                    detect_and_plot(plot_range_df, file_name_range)
                    logger.info(f"PDF des données de la plage [{start_idx}:{end_idx}] généré : {file_name_range}")
            except ValueError:
                logger.warning("Veuillez entrer des indices numériques valides.")
        else:
            logger.warning("Aucune donnée disponible pour générer le graphique.")
    elif user_input == 'EXIT':
        logger.info("Programme en cours de sortie.")
        if monitoring:
            monitoring = False
            stop_event.set()
            monitoring_thread.join()
        for f in glob.glob("30SecPlotting/*.pdf"):
            os.remove(f)
        break
    else:
        logger.warning("Entrée non valide. Veuillez entrer 'ON', 'OFF', 'PLOT_RANGE' ou 'EXIT'.")
