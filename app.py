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
df = df[['Airline', 'Source', 'Destination', 'Duration', 'Total_Stops',
         'Dep_Time', 'Arrival_Time', 'Date_of_Journey', 'Price']]
df.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales',
              'heure_depart', 'heure_arrivee', 'date_voyage', 'prix']

def convertir_duree(duree):
    heures = 0; minutes = 0
    if 'h' in str(duree): heures = int(str(duree).split('h')[0])
    if 'm' in str(duree): minutes = int(str(duree).split('h')[-1].replace('m', '').strip())
    return heures * 60 + minutes

df.loc[:, 'duree'] = df['duree'].apply(convertir_duree)
df['heure_depart_num'] = pd.to_datetime(df['heure_depart'], format='%H:%M').dt.hour
df['heure_arrivee_num'] = df['heure_arrivee'].astype(str).str.extract(r'(\d+):(\d+)')[0].astype(float)
df['date_voyage'] = pd.to_datetime(df['date_voyage'], format='%d/%m/%Y')
df['mois_voyage'] = df['date_voyage'].dt.month
df['jour_semaine'] = df['date_voyage'].dt.dayofweek
df = df.drop(columns=['date_voyage'])
df['escales_num'] = df['escales'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
df = df.dropna(subset=['prix', 'compagnie'])

encode_compagnie = LabelEncoder()
encode_depart = LabelEncoder()
encode_arrivee = LabelEncoder()
df['compagnie_enc'] = encode_compagnie.fit_transform(df['compagnie'].astype(str))
df['depart_enc'] = encode_depart.fit_transform(df['depart'].astype(str))
df['arrivee_enc'] = encode_arrivee.fit_transform(df['arrivee'].astype(str))
df['mois_nom'] = df['mois_voyage'].map({3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin'})

def obtenir_tranche(heure):
    if pd.isna(heure): return 'Inconnu'
    h = int(str(heure).split(':')[0])
    if h < 6: return 'Nuit'
    elif h < 12: return 'Matin'
    elif h < 18: return 'Après-midi'
    else: return 'Soir'

df['tranche'] = df['heure_depart'].apply(obtenir_tranche)

# ── Chargement ou réentraînement du modèle ───────────────────────────
if os.path.exists('modele.joblib'):
    modele = joblib.load('modele.joblib')
else:
    donnees_train = df[['compagnie_enc', 'depart_enc', 'arrivee_enc', 'duree',
                         'escales_num', 'heure_depart_num', 'heure_arrivee_num',
                         'mois_voyage', 'jour_semaine']].copy()
    donnees_train.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales',
                              'heure_depart', 'heure_arrivee', 'mois_voyage', 'jour_semaine']
    for col in donnees_train.columns:
        donnees_train[col] = donnees_train[col].fillna(donnees_train[col].mode()[0])
    modele = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    modele.fit(donnees_train, df['prix'])

app = dash.Dash(__name__)
server = app.server

COULEURS = {
    'fond': '#d4d5d6', 'surface': '#f8fafc', 'accent': '#2563eb',
    'texte': '#0f172a', 'muted': '#1e293b', 'ambre': '#d97706',
}

# ── Libellés des tranches : simples dans le filtre, avec plages sur le graphique
TRANCHES_CLES = ['Nuit', 'Matin', 'Après-midi', 'Soir']
TRANCHES_GRAPHIQUE = ['Nuit<br>(00h–06h)', 'Matin<br>(06h–12h)', 'Après-midi<br>(12h–18h)', 'Soir<br>(18h–00h)']

destinations_liste = sorted(df['arrivee'].dropna().unique())

app.layout = html.Div(
    style={'backgroundColor': COULEURS['fond'], 'padding': '20px', 'fontFamily': 'Inter, sans-serif'},
    children=[
        html.H1("Les meilleurs moments pour voyager en Inde en 2019.",
                style={'color': COULEURS['texte'], 'fontSize': '18px', 'marginBottom': '4px'}),
        html.P("Analyse des prix réels · Modèle Random Forest · R² = 0.72",
               style={'color': COULEURS['muted'], 'fontSize': '12px', 'marginBottom': '16px'}),

        # ── Filtres ──────────────────────────────────────────────────
        html.Div(
            style={'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px',
                   'marginBottom': '12px', 'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'},
            children=[
                html.Div([
                    html.Label("Mois", style={'fontSize': '10px', 'color': COULEURS['muted'], 'fontWeight': '600'}),
                    dcc.Dropdown(id='filtre-mois',
                        options=[{'label': 'Tous', 'value': 'Tous'}] + [{'label': m, 'value': m} for m in ['Mars', 'Avril', 'Mai', 'Juin']],
                        value='Tous', clearable=False),
                ], style={'flex': 1, 'minWidth': '120px'}),
                html.Div([
                    html.Label("Compagnie", style={'fontSize': '10px', 'color': COULEURS['muted'], 'fontWeight': '600'}),
                    dcc.Dropdown(id='filtre-compagnie',
                        options=[{'label': 'Toutes', 'value': 'Toutes'}] + [{'label': c, 'value': c} for c in sorted(df['compagnie'].unique())],
                        value='Toutes', clearable=False),
                ], style={'flex': 1, 'minWidth': '120px'}),
                html.Div([
                    # Filtre : noms simples sans plages horaires
                    html.Label("Moment de la journée", style={'fontSize': '10px', 'color': COULEURS['muted'], 'fontWeight': '600'}),
                    dcc.Dropdown(id='filtre-tranche',
                        options=[{'label': 'Tous', 'value': 'Tous'}] + [{'label': t, 'value': t} for t in TRANCHES_CLES],
                        value='Tous', clearable=False),
                ], style={'flex': 1, 'minWidth': '120px'}),
                html.Div([
                    html.Label("Destination", style={'fontSize': '10px', 'color': COULEURS['muted'], 'fontWeight': '600'}),
                    dcc.Dropdown(id='filtre-destination',
                        options=[{'label': 'Toutes', 'value': 'Toutes'}] + [{'label': d, 'value': d} for d in destinations_liste],
                        value='Toutes', clearable=False),
                ], style={'flex': 1, 'minWidth': '120px'}),
            ]
        ),

        # ── KPIs ─────────────────────────────────────────────────────
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '12px', 'flexWrap': 'wrap'}, children=[
            html.Div(id='kpi-vols', style={'flex': 1, 'minWidth': '120px', 'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px'}),
            html.Div(id='kpi-reel', style={'flex': 1, 'minWidth': '120px', 'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px'}),
            html.Div(id='kpi-predit', style={'flex': 1, 'minWidth': '120px', 'backgroundColor': COULEURS['surface'], 'padding': '14px', 'borderRadius': '12px'}),
        ]),

        # ── Graphiques ───────────────────────────────────────────────
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px', 'flexWrap': 'wrap'}, children=[
            html.Div(dcc.Graph(id='graph-tranche'), style={'flex': 1, 'minWidth': '300px', 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
            html.Div(dcc.Graph(id='graph-mois'), style={'flex': 1, 'minWidth': '300px', 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
        ]),
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px', 'flexWrap': 'wrap'}, children=[
            html.Div(dcc.Graph(id='graph-compagnie'), style={'flex': 1, 'minWidth': '300px', 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
            html.Div(dcc.Graph(id='graph-escales'), style={'flex': 1, 'minWidth': '300px', 'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
        ]),
        html.Div(dcc.Graph(id='graph-destinations'), style={'backgroundColor': COULEURS['surface'], 'borderRadius': '12px', 'padding': '10px'}),
    ]
)


@app.callback(
    Output('kpi-vols', 'children'), Output('kpi-reel', 'children'), Output('kpi-predit', 'children'),
    Input('filtre-mois', 'value'), Input('filtre-compagnie', 'value'),
    Input('filtre-tranche', 'value'), Input('filtre-destination', 'value'),
)
def mettre_a_jour_kpis(mois, compagnie, tranche, destination):
    dff = df.copy()
    if mois != 'Tous': dff = dff[dff['mois_nom'] == mois]
    if compagnie != 'Toutes': dff = dff[dff['compagnie'] == compagnie]
    if tranche != 'Tous': dff = dff[dff['tranche'] == tranche]
    if destination != 'Toutes': dff = dff[dff['arrivee'] == destination]

    nb_vols = len(dff)
    prix_reel = round(dff['prix'].mean()) if nb_vols > 0 else 0
    if nb_vols > 0:
        dff_pred = dff[['compagnie_enc', 'depart_enc', 'arrivee_enc', 'duree', 'escales_num',
                         'heure_depart_num', 'heure_arrivee_num', 'mois_voyage', 'jour_semaine']].copy()
        dff_pred.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales',
                             'heure_depart', 'heure_arrivee', 'mois_voyage', 'jour_semaine']
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


@app.callback(
    Output('graph-tranche', 'figure'), Output('graph-mois', 'figure'), Output('graph-compagnie', 'figure'),
    Output('graph-escales', 'figure'), Output('graph-destinations', 'figure'),
    Input('filtre-mois', 'value'), Input('filtre-compagnie', 'value'),
    Input('filtre-tranche', 'value'), Input('filtre-destination', 'value'),
)
def mettre_a_jour_graphiques(mois, compagnie, tranche, destination):

    def filtrer(exclure=None):
        """Applique tous les filtres sauf la dimension du graphique lui-même."""
        d = df.copy()
        if exclure != 'mois' and mois != 'Tous': d = d[d['mois_nom'] == mois]
        if exclure != 'compagnie' and compagnie != 'Toutes': d = d[d['compagnie'] == compagnie]
        if exclure != 'tranche' and tranche != 'Tous': d = d[d['tranche'] == tranche]
        if exclure != 'destination' and destination != 'Toutes': d = d[d['arrivee'] == destination]
        return d

    def prix_predit_moyen(subset):
        if len(subset) == 0: return 0
        p = subset[['compagnie_enc', 'depart_enc', 'arrivee_enc', 'duree', 'escales_num',
                     'heure_depart_num', 'heure_arrivee_num', 'mois_voyage', 'jour_semaine']].copy()
        p.columns = ['compagnie', 'depart', 'arrivee', 'duree', 'escales',
                     'heure_depart', 'heure_arrivee', 'mois_voyage', 'jour_semaine']
        p = p.fillna(p.mode().iloc[0])
        return round(modele.predict(p).mean())

    def val(v):
        return round(v) if isinstance(v, float) and not np.isnan(v) else (int(v) if not isinstance(v, float) else 0)

    cfg = dict(paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc',
               margin=dict(t=40, b=10, l=10, r=10), title_font_size=12,
               legend=dict(font=dict(size=10)))

    # ── G1 : barres verticales groupées — tranche (exclut filtre tranche) ──
    dff_tr = filtrer(exclure='tranche')
    r_tr = [val(dff_tr[dff_tr['tranche'] == t]['prix'].mean()) for t in TRANCHES_CLES]
    p_tr = [prix_predit_moyen(dff_tr[dff_tr['tranche'] == t]) for t in TRANCHES_CLES]
    # Surbrillance : couleur pleine sur la tranche sélectionnée, pâle sur les autres
    cr_tr = ['#1e40af' if tranche == 'Tous' or t == tranche else '#bfdbfe' for t in TRANCHES_CLES]
    cp_tr = ['#b45309' if tranche == 'Tous' or t == tranche else '#fde68a' for t in TRANCHES_CLES]

    fig_tr = go.Figure()
    fig_tr.add_trace(go.Bar(x=TRANCHES_GRAPHIQUE, y=r_tr, name='Réel', marker_color=cr_tr,
        text=[f"{v:,} ₹".replace(',', ' ') for v in r_tr], textposition='outside'))
    fig_tr.add_trace(go.Bar(x=TRANCHES_GRAPHIQUE, y=p_tr, name='Prédit', marker_color=cp_tr,
        text=[f"{v:,} ₹".replace(',', ' ') for v in p_tr], textposition='outside'))
    fig_tr.update_layout(title='Prix des billets selon le moment de la journée.',
        barmode='group', yaxis=dict(gridcolor='#f1f5f9', title='Prix moyen (₹)'), **cfg)

    # ── G2 : courbe + aire — mois (exclut filtre mois) ──────────────
    dff_mo = filtrer(exclure='mois')
    mois_ordre = ['Mars', 'Avril', 'Mai', 'Juin']
    r_mo = [val(dff_mo[dff_mo['mois_nom'] == m]['prix'].mean()) for m in mois_ordre]
    p_mo = [prix_predit_moyen(dff_mo[dff_mo['mois_nom'] == m]) for m in mois_ordre]
    taille_m = [14 if mois == 'Tous' or m == mois else 7 for m in mois_ordre]
    cm_r = ['#7c3aed' if mois == 'Tous' or m == mois else '#c4b5fd' for m in mois_ordre]
    cm_p = ['#b45309' if mois == 'Tous' or m == mois else '#fde68a' for m in mois_ordre]

    fig_mo = go.Figure()
    fig_mo.add_trace(go.Scatter(x=mois_ordre, y=r_mo, mode='lines+markers', name='Réel',
        fill='tozeroy', fillcolor='rgba(124,58,237,0.08)', line=dict(color='#7c3aed', width=2),
        marker=dict(size=taille_m, color=cm_r, line=dict(width=2, color='#7c3aed'))))
    fig_mo.add_trace(go.Scatter(x=mois_ordre, y=p_mo, mode='lines+markers', name='Prédit',
        fill='tozeroy', fillcolor='rgba(217,119,6,0.07)', line=dict(color='#d97706', width=2, dash='dash'),
        marker=dict(size=taille_m, color=cm_p, line=dict(width=2, color='#d97706'))))
    fig_mo.update_layout(title='Évolution des prix au fil des mois.',
        yaxis=dict(gridcolor='#f1f5f9', title='Prix moyen (₹)'), **cfg)

    # ── G3 : barres horizontales groupées — compagnie (exclut filtre compagnie) ──
    dff_co = filtrer(exclure='compagnie')
    compagnies = sorted(df['compagnie'].unique())
    r_co = [val(dff_co[dff_co['compagnie'] == c]['prix'].mean()) for c in compagnies]
    p_co = [prix_predit_moyen(dff_co[dff_co['compagnie'] == c]) for c in compagnies]
    cr_co = ['#059669' if compagnie == 'Toutes' or c == compagnie else '#a7f3d0' for c in compagnies]
    cp_co = ['#b45309' if compagnie == 'Toutes' or c == compagnie else '#fde68a' for c in compagnies]

    fig_co = go.Figure()
    fig_co.add_trace(go.Bar(x=r_co, y=compagnies, orientation='h', name='Réel', marker_color=cr_co,
        text=[f"{v:,} ₹".replace(',', ' ') for v in r_co], textposition='outside'))
    fig_co.add_trace(go.Bar(x=p_co, y=compagnies, orientation='h', name='Prédit', marker_color=cp_co,
        text=[f"{v:,} ₹".replace(',', ' ') for v in p_co], textposition='outside'))
    fig_co.update_layout(title='Prix moyen par compagnie aérienne.',
        barmode='group', xaxis=dict(gridcolor='#f1f5f9', title='Prix moyen (₹)'),
        margin=dict(t=40, b=10, l=10, r=60), paper_bgcolor='#f8fafc',
        plot_bgcolor='#f8fafc', title_font_size=12, legend=dict(font=dict(size=10)))

    # ── G4 : courbe + aire — escales (utilise tous les filtres actifs) ──
    dff_esc = filtrer()
    escales_vals = [0, 1, 2, 3]
    escales_labels = ['0 escale', '1 escale', '2 escales', '3 escales']
    r_esc = [val(dff_esc[dff_esc['escales_num'] == e]['prix'].mean()) for e in escales_vals]
    p_esc = [prix_predit_moyen(dff_esc[dff_esc['escales_num'] == e]) for e in escales_vals]

    fig_esc = go.Figure()
    fig_esc.add_trace(go.Scatter(x=escales_labels, y=r_esc, mode='lines+markers', name='Prix réels',
        fill='tozeroy', fillcolor='rgba(37,99,235,0.1)', line=dict(color='#2563eb', width=2)))
    fig_esc.add_trace(go.Scatter(x=escales_labels, y=p_esc, mode='lines+markers', name='Prix prédits',
        fill='tozeroy', fillcolor='rgba(217,119,6,0.07)', line=dict(color='#d97706', width=2, dash='dash')))
    fig_esc.update_layout(title='Impact du nombre d\'escales sur le prix.',
        yaxis=dict(gridcolor='#f1f5f9', title='Prix moyen (₹)'), **cfg)

    # ── G5 : barres horizontales groupées — destinations (exclut filtre destination) ──
    dff_de = filtrer(exclure='destination')
    toutes_dest = sorted(df['arrivee'].dropna().unique())
    r_de = [val(dff_de[dff_de['arrivee'] == d]['prix'].mean()) for d in toutes_dest]
    p_de = [prix_predit_moyen(dff_de[dff_de['arrivee'] == d]) for d in toutes_dest]
    cr_de = ['#2563eb' if destination == 'Toutes' or d == destination else '#bfdbfe' for d in toutes_dest]
    cp_de = ['#b45309' if destination == 'Toutes' or d == destination else '#fde68a' for d in toutes_dest]

    fig_de = go.Figure()
    fig_de.add_trace(go.Bar(x=r_de, y=toutes_dest, orientation='h', name='Réel', marker_color=cr_de,
        text=[f"{v:,} ₹".replace(',', ' ') for v in r_de], textposition='outside'))
    fig_de.add_trace(go.Bar(x=p_de, y=toutes_dest, orientation='h', name='Prédit', marker_color=cp_de,
        text=[f"{v:,} ₹".replace(',', ' ') for v in p_de], textposition='outside'))
    fig_de.update_layout(title='Prix moyen selon la destination.',
        barmode='group', xaxis=dict(gridcolor='#f1f5f9', title='Prix moyen (₹)'),
        margin=dict(t=40, b=10, l=10, r=60), paper_bgcolor='#f8fafc',
        plot_bgcolor='#f8fafc', title_font_size=12, legend=dict(font=dict(size=10)))

    return fig_tr, fig_mo, fig_co, fig_esc, fig_de


if __name__ == '__main__':
    app.run(debug=True)
