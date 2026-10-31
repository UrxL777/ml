# Librairies pour le dashboard web
import os
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ── Chargement des données ───────────────────────────────────────────
df = pd.read_csv('planes.csv')

# ── Sélection des colonnes utiles uniquement ─────────────────────────
df = df[['Airline', 'Source', 'Destination', 'Duration', 'Total_Stops',
         'Dep_Time', 'Arrival_Time', 'Date_of_Journey', 'Price']]
df.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales',
              'heure_depart', 'heure_arrivee', 'date_voyage', 'prix']

# ── Conversion de la durée en minutes ────────────────────────────────
def convertir_duree(duree):
    heures = 0; minutes = 0
    if 'h' in str(duree): heures = int(str(duree).split('h')[0])
    if 'm' in str(duree): minutes = int(str(duree).split('h')[-1].replace('m', '').strip())
    return heures * 60 + minutes

df['duree'] = df['duree'].apply(convertir_duree)

# ── Extraction des heures numériques ─────────────────────────────────
df['heure_depart_num'] = pd.to_datetime(df['heure_depart'], format='%H:%M').dt.hour
df['heure_arrivee_num'] = df['heure_arrivee'].astype(str).str.extract(r'(\d+):(\d+)')[0].astype(float)

# ── Extraction du mois et du jour de la semaine ───────────────────────
df['date_voyage'] = pd.to_datetime(df['date_voyage'], format='%d/%m/%Y')
df['mois_voyage'] = df['date_voyage'].dt.month
df['jour_semaine'] = df['date_voyage'].dt.dayofweek
df = df.drop(columns=['date_voyage'])

# ── Conversion des escales en chiffres ───────────────────────────────
df['escales_num'] = df['escales'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

# ── Suppression des lignes sans prix ni compagnie ────────────────────
df = df.dropna(subset=['prix'])
df = df.dropna(subset=['compagnie'])

# ── Encodage des colonnes texte en chiffres ───────────────────────────
encode_compagnie = LabelEncoder()
encode_depart = LabelEncoder()
encode_arrivee = LabelEncoder()
df['compagnie_enc'] = encode_compagnie.fit_transform(df['compagnie'].astype(str))
df['depart_enc'] = encode_depart.fit_transform(df['depart'].astype(str))
df['arrivee_enc'] = encode_arrivee.fit_transform(df['arrivee'].astype(str))

# ── Ajout du nom du mois en français ─────────────────────────────────
df['mois_nom'] = df['mois_voyage'].map({3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin'})

# ── Fonction pour déterminer la tranche horaire ───────────────────────
def obtenir_tranche(heure):
    if pd.isna(heure): return 'Inconnu'
    h = int(str(heure).split(':')[0])
    if h < 6: return 'Nuit'
    elif h < 12: return 'Matin'
    elif h < 18: return 'Après-midi'
    else: return 'Soir'

df['tranche'] = df['heure_depart'].apply(obtenir_tranche)

# ── Chargement ou réentraînement du modèle ───────────────────────────
# Sur Render, modele.joblib n'existe pas → on réentraîne automatiquement
if os.path.exists('modele.joblib'):
    modele = joblib.load('modele.joblib')
else:
    # Préparation des features pour l'entraînement
    X_train = df[['compagnie_enc', 'depart_enc', 'arrivee_enc', 'duree',
                  'escales_num', 'heure_depart_num', 'heure_arrivee_num',
                  'mois_voyage', 'jour_semaine']].copy()
    X_train.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales',
                       'heure_depart', 'heure_arrivee', 'mois_voyage', 'jour_semaine']
    X_train = X_train.fillna(X_train.mode().iloc[0])
    Y_train = df['prix']
    # Entraînement du modèle RandomForest
    modele = RandomForestRegressor(n_estimators=100, random_state=42)
    modele.fit(X_train, Y_train)

# ── Initialisation de l'application Dash ─────────────────────────────
app = dash.Dash(__name__)
server = app.server  # nécessaire pour gunicorn sur Render

# ── Palette de couleurs ───────────────────────────────────────────────
COULEURS = {
    'fond': '#94a3b8', 'surface': '#f8fafc', 'accent': '#2563eb',
    'texte': '#0f172a', 'muted': '#1e293b', 'vert': '#16a34a',
    'ambre': '#d97706', 'bordure': '#e2e8f0',
}

# ── Mise en page du dashboard ─────────────────────────────────────────
app.layout = html.Div(
    style={'backgroundColor': COULEURS['fond'], 'padding': '20px', 'fontFamily': 'Inter, sans-serif'},
    children=[
        html.H1("Quel est le meilleur moment pour voyager en Inde en 2019 ?",
                style={'color': COULEURS['texte'], 'fontSize': '18px', 'marginBottom': '4px'}),
        html.P("Analyse des prix réels · Modèle Random Forest · R² = 0.72",
               style={'color': COULEURS['muted'], 'fontSize': '12px', 'marginBottom': '16px'}),
        # ── Filtres ──
        html.Div(
            style={'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px', 'marginBottom': '12px', 'display': 'flex', 'gap': '20px'},
            children=[
                html.Div([
                    html.Label("Mois", style={'fontSize': '10px', 'color': COULEURS['muted'], 'fontWeight': '600'}),
                    dcc.Dropdown(id='filtre-mois', options=[{'label': 'Tous', 'value': 'Tous'}] + [{'label': m, 'value': m} for m in ['Mars', 'Avril', 'Mai', 'Juin']], value='Tous', clearable=False),
                ], style={'flex': 1}),
                html.Div([
                    html.Label("Compagnie", style={'fontSize': '10px', 'color': COULEURS['muted'], 'fontWeight': '600'}),
                    dcc.Dropdown(id='filtre-compagnie', options=[{'label': 'Toutes', 'value': 'Toutes'}] + [{'label': c, 'value': c} for c in sorted(df['compagnie'].unique())], value='Toutes', clearable=False),
                ], style={'flex': 1}),
                html.Div([
                    html.Label("Moment de la journée", style={'fontSize': '10px', 'color': COULEURS['muted'], 'fontWeight': '600'}),
                    dcc.Dropdown(id='filtre-tranche', options=[{'label': 'Tous', 'value': 'Tous'}] + [{'label': t, 'value': t} for t in ['Nuit', 'Matin', 'Après-midi', 'Soir']], value='Tous', clearable=False),
                ], style={'flex': 1}),
            ]
        ),
        # ── KPIs ──
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '12px'}, children=[
            html.Div(id='kpi-vols', style={'flex': 1, 'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px'}),
            html.Div(id='kpi-reel', style={'flex': 1, 'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px'}),
            html.Div(id='kpi-predit', style={'flex': 1, 'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px'}),
        ]),
        # ── Graphiques ligne 1 ──
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}, children=[
            html.Div(dcc.Graph(id='graph-tranche'), style={'flex': 1, 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
            html.Div(dcc.Graph(id='graph-mois'), style={'flex': 1, 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
        ]),
        # ── Graphiques ligne 2 ──
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}, children=[
            html.Div(dcc.Graph(id='graph-compagnie'), style={'flex': 1, 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
            html.Div(dcc.Graph(id='graph-escales'), style={'flex': 1, 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
        ]),
        # ── Graphique destinations ──
        html.Div(dcc.Graph(id='graph-destinations'), style={'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
    ]
)

# ── Callback pour les KPIs ────────────────────────────────────────────
@app.callback(
    Output('kpi-vols', 'children'), Output('kpi-reel', 'children'), Output('kpi-predit', 'children'),
    Input('filtre-mois', 'value'), Input('filtre-compagnie', 'value'), Input('filtre-tranche', 'value'),
)
def mettre_a_jour_kpis(mois, compagnie, tranche):
    dff = df.copy()
    if mois != 'Tous': dff = dff[dff['mois_nom'] == mois]
    if compagnie != 'Toutes': dff = dff[dff['compagnie'] == compagnie]
    if tranche != 'Tous': dff = dff[dff['tranche'] == tranche]
    nb_vols = len(dff)
    prix_reel = round(dff['prix'].mean()) if nb_vols > 0 else 0
    if nb_vols > 0:
        dff_pred = dff[['compagnie_enc', 'depart_enc', 'arrivee_enc', 'duree', 'escales_num', 'heure_depart_num', 'heure_arrivee_num', 'mois_voyage', 'jour_semaine']].copy()
        dff_pred.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales', 'heure_depart', 'heure_arrivee', 'mois_voyage', 'jour_semaine']
        dff_pred = dff_pred.fillna(dff_pred.mode().iloc[0])
        prix_predit = round(modele.predict(dff_pred).mean())
    else:
        prix_predit = 0
    style_lbl = {'fontSize': '10px', 'textTransform': 'uppercase', 'color': COULEURS['muted'], 'fontWeight': '600'}
    style_sub = {'fontSize': '10px', 'color': COULEURS['muted'], 'marginTop': '2px'}
    kpi_vols = html.Div([html.Div("Nombre de vols", style=style_lbl), html.Div(f"{nb_vols:,}".replace(',', ' '), style={'fontSize': '22px', 'fontWeight': '700', 'marginTop': '5px'}), html.Div("dans la sélection", style=style_sub)])
    kpi_reel = html.Div([html.Div("Prix réel moyen", style=style_lbl), html.Div(f"{prix_reel:,} ₹".replace(',', ' '), style={'fontSize': '22px', 'fontWeight': '700', 'color': COULEURS['accent'], 'marginTop': '5px'}), html.Div("roupies indiennes", style=style_sub)])
    kpi_predit = html.Div([html.Div("Prix prédit moyen", style=style_lbl), html.Div(f"{prix_predit:,} ₹".replace(',', ' '), style={'fontSize': '22px', 'fontWeight': '700', 'color': COULEURS['ambre'], 'marginTop': '5px'}), html.Div("estimé par le modèle", style=style_sub)])
    return kpi_vols, kpi_reel, kpi_predit

# ── Callback pour les 5 graphiques ───────────────────────────────────
@app.callback(
    Output('graph-tranche', 'figure'), Output('graph-mois', 'figure'), Output('graph-compagnie', 'figure'),
    Output('graph-escales', 'figure'), Output('graph-destinations', 'figure'),
    Input('filtre-mois', 'value'), Input('filtre-compagnie', 'value'), Input('filtre-tranche', 'value'),
)
def mettre_a_jour_graphiques(mois, compagnie, tranche):
    dff = df.copy()
    if mois != 'Tous': dff = dff[dff['mois_nom'] == mois]
    if compagnie != 'Toutes': dff = dff[dff['compagnie'] == compagnie]
    if tranche != 'Tous': dff = dff[dff['tranche'] == tranche]

    # Fonction pour calculer le prix prédit moyen d'un sous-ensemble
    def prix_predit_moyen(subset):
        if len(subset) == 0: return 0
        pred_data = subset[['compagnie_enc', 'depart_enc', 'arrivee_enc', 'duree', 'escales_num', 'heure_depart_num', 'heure_arrivee_num', 'mois_voyage', 'jour_semaine']].copy()
        pred_data.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales', 'heure_depart', 'heure_arrivee', 'mois_voyage', 'jour_semaine']
        pred_data = pred_data.fillna(pred_data.mode().iloc[0])
        return round(modele.predict(pred_data).mean())

    # 1. DONUT — tranches horaires
    tranches = ['Nuit', 'Matin', 'Après-midi', 'Soir']
    prix_tranches = [dff[dff['tranche'] == t]['prix'].mean() for t in tranches]
    fig_tranche = go.Figure(go.Pie(labels=tranches, values=[round(p) if not np.isnan(p) else 0 for p in prix_tranches], hole=0.5, marker_colors=['#1e40af', '#2563eb', '#60a5fa', '#93c5fd']))
    fig_tranche.update_layout(title='À quel moment de la journée voyager moins cher ?', title_font_size=12, margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc')

    # 2. BARRES VERTICALES — mois (violet)
    mois_ordre = ['Mars', 'Avril', 'Mai', 'Juin']
    prix_mois = [dff[dff['mois_nom'] == m]['prix'].mean() for m in mois_ordre]
    couleurs_mois = ['#7c3aed' if mois == 'Tous' or m == mois else '#ddd6fe' for m in mois_ordre]
    fig_mois = go.Figure(go.Bar(x=mois_ordre, y=[round(p) if not np.isnan(p) else 0 for p in prix_mois], marker_color=couleurs_mois, text=[f"{round(p):,} ₹".replace(',', ' ') if not np.isnan(p) else '' for p in prix_mois], textposition='outside'))
    fig_mois.update_layout(title='Dans quel mois voyager moins cher ?', title_font_size=12, margin=dict(t=40, b=10, l=10, r=10), yaxis_title='Prix moyen (₹)', paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc', yaxis=dict(gridcolor='#f1f5f9'))

    # 3. BARRES HORIZONTALES — compagnies (vert)
    compagnies = sorted(dff['compagnie'].unique())
    prix_comp = [dff[dff['compagnie'] == c]['prix'].mean() for c in compagnies]
    couleurs_comp = ['#059669' if compagnie == 'Toutes' or c == compagnie else '#6ee7b7' for c in compagnies]
    fig_comp = go.Figure(go.Bar(x=[round(p) if not np.isnan(p) else 0 for p in prix_comp], y=compagnies, orientation='h', marker_color=couleurs_comp, text=[f"{round(p):,} ₹".replace(',', ' ') if not np.isnan(p) else '' for p in prix_comp], textposition='outside'))
    fig_comp.update_layout(title='Quelle compagnie choisir ?', title_font_size=12, margin=dict(t=40, b=10, l=10, r=10), xaxis_title='Prix moyen (₹)', paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc', xaxis=dict(gridcolor='#f1f5f9'))

    # 4. AREA CHART — escales (bleu + orange)
    escales_vals = [0, 1, 2, 3]
    escales_labels = ['0 escale', '1 escale', '2 escales', '3 escales']
    prix_esc_reel = [dff[dff['escales_num'] == e]['prix'].mean() for e in escales_vals]
    prix_esc_pred = [prix_predit_moyen(dff[dff['escales_num'] == e]) for e in escales_vals]
    fig_esc = go.Figure()
    fig_esc.add_trace(go.Scatter(x=escales_labels, y=[round(p) if not np.isnan(p) else 0 for p in prix_esc_reel], mode='lines+markers', fill='tozeroy', name='Prix réels', line=dict(color='#2563eb', width=2), fillcolor='rgba(37,99,235,0.1)'))
    fig_esc.add_trace(go.Scatter(x=escales_labels, y=prix_esc_pred, mode='lines+markers', fill='tozeroy', name='Prix prédits', line=dict(color='#d97706', width=2, dash='dash'), fillcolor='rgba(217,119,6,0.07)'))
    fig_esc.update_layout(title='Les escales font-elles augmenter le prix ?', title_font_size=12, margin=dict(t=40, b=10, l=10, r=10), yaxis_title='Prix moyen (₹)', paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc', yaxis=dict(gridcolor='#f1f5f9'), legend=dict(font=dict(size=10)))

    # 5. SCATTER — destinations (bleu + orange)
    destinations = sorted(dff['arrivee'].dropna().unique())
    scatter_x, scatter_y_reel, scatter_y_pred = [], [], []
    for dest in destinations:
        subset = dff[dff['arrivee'] == dest]
        if len(subset) > 0:
            scatter_x.append(dest)
            scatter_y_reel.append(round(subset['prix'].mean()))
            scatter_y_pred.append(prix_predit_moyen(subset))
    fig_dest = go.Figure()
    fig_dest.add_trace(go.Scatter(x=scatter_x, y=scatter_y_reel, mode='markers', name='Prix réels', marker=dict(color='#2563eb', size=12)))
    fig_dest.add_trace(go.Scatter(x=scatter_x, y=scatter_y_pred, mode='markers', name='Prix prédits', marker=dict(color='#d97706', size=12, symbol='diamond')))
    fig_dest.update_layout(title='Vers quelle destination les billets sont-ils les moins chers ?', title_font_size=12, margin=dict(t=40, b=10, l=10, r=10), yaxis_title='Prix moyen (₹)', paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc', yaxis=dict(gridcolor='#f1f5f9'), xaxis=dict(gridcolor='#f1f5f9'), legend=dict(font=dict(size=10)))

    return fig_tranche, fig_mois, fig_comp, fig_esc, fig_dest

# ── Lancement de l'application ────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)