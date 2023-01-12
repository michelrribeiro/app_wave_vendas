# Aplicação para avaliar vendas, fazer recomendações e fazer previsões
#-------------------------------------------------------------------------------------------------------
# Imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly import io as pio
from statsmodels.tsa.statespace.sarimax import SARIMAX
from h2o_wave import Q, app, data, main, ui
#-------------------------------------------------------------------------------------------------------
# Carregando os dataframes necessários:
df_ticket = pd.read_csv('./../data/df_ticket.csv', index_col=0)
df_prev = pd.read_csv('./../data/df_prev.csv', parse_dates=['Order Date'], index_col=0)
df_cats = pd.read_csv('./../data/df_cats.csv', index_col=0)

# Lista de subcategorias:
def lista_subcat(seg):
    prop_vendas_nome = seg.lower().replace(' ', '_') + '_prop_vendas.csv'
    prop_vendas = pd.read_csv(f'./../data/{prop_vendas_nome}', index_col=0)
    nomes = [n for n in prop_vendas['Sub-Category'].unique()]
    return np.sort(nomes)

# Lista de produtos com base no segmento e subcategoria:
def lista_produtos(seg, subcategoria):
    prop_vendas_nome = seg.lower().replace(' ', '_') + '_prop_vendas.csv'
    prop_vendas = pd.read_csv(f'./../data/{prop_vendas_nome}', index_col=0)
    nomes = prop_vendas[prop_vendas['Sub-Category']==subcategoria]['Product Name'].to_list()
    return np.sort(nomes)

# Gráfico com total de vendas por segmento:
def total_seg(df_ticket):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_ticket.segmento.unique(), y=df_ticket['total_vendas'], 
                         text=[(str(round(n/10e2))+'K') if n <10e5 else (str(round(n/10e5, 3))+'Mi') for n in df_ticket['total_vendas']], 
                         textposition='auto'))

    fig.update_layout(title_text='Total de vendas por segmento', 
                      title_font_color='black', title_x=0.5, title_y=0.945, title_font_size=10, 
                      margin=dict(l=0, r=0, b=0, t=30))
    
    fig.update_xaxes(tickfont_size=9, tickfont_color='black')
    fig.update_yaxes(visible=False)
    
    return (pio.to_html(fig, include_plotlyjs='cdn', validate=False))

# Gráfico com ticket médio (total de vendas/núm de ordens de compra) para cada segmento:
def ticket_seg(df_ticket):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_ticket.segmento, y=df_ticket['ticket_medio'], 
                         text=df_ticket['ticket_medio'], 
                         textposition='auto'))

    fig.update_layout(title_text='Ticket médio por segmento', 
                      title_font_color='black', title_x=0.5, title_y=0.945, title_font_size=10, 
                      margin=dict(l=0, r=0, b=0, t=30))
    
    fig.update_xaxes(tickfont_size=9, tickfont_color='black')
    fig.update_yaxes(visible=False)
    
    return (pio.to_html(fig, include_plotlyjs='cdn', validate=False))

# Gráfico com total de vendas para cada categoria a depender do segmento:
def total_cat(seg, df_cats):
    df_cats = df_cats[df_cats.Segment==seg]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_cats.groupby('Category', as_index=False).agg({'Sales': 'sum'})['Category'], 
                         y=df_cats.groupby('Category', as_index=False).agg({'Sales': 'sum'})['Sales'], 
                         text=[(str(round(n/10e2))+'K') for n in df_cats.groupby(
                             'Category', as_index=False).agg({'Sales': 'sum'})['Sales']], 
                         textposition='auto'))

    fig.update_layout(title_text=f'Total de vendas por categoria <br> no segmento {seg}', 
                      title_font_color='black', title_x=0.5, title_y=0.945, title_font_size=10, 
                      margin=dict(l=0, r=0, b=0, t=30))
    
    fig.update_xaxes(tickfont_size=9, tickfont_color='black')
    fig.update_yaxes(visible=False)
  
    return (pio.to_html(fig, include_plotlyjs='cdn', validate=False))

# Gráfico com vendas ano a ano para cada categoria a depender do segmento:
def total_cat_ano(seg, df_cats):
    df_cats = df_cats[df_cats.Segment==seg]
    cats = df_cats.Category.unique()

    fig = go.Figure()
    for n in cats:
        fig.add_trace(go.Scatter(x=[str(n) for n in df_cats.year.unique()], 
                                 y=df_cats[df_cats.Category==n]['Sales'],
                                 name = n))

    fig.update_layout(title_text=f'Total de vendas por categoria no segmento Consumer', 
                      title_font_color='black', title_x=0.5, title_font_size=10, 
                      legend={'orientation':'h', 
                              'y':1, 'yanchor': 'top', 
                              'x':0, 'xanchor': 'left', 
                              'font':{'size':8, 'color':'black'}, 'itemclick':'toggle'}, 
                      margin=dict(l=0, r=0, b=0, t=15))
    
    fig.update_xaxes(tickfont_size=9, tickfont_color='black')
    fig.update_yaxes(visible=False)

    return (pio.to_html(fig, include_plotlyjs='cdn', validate=False))

# Função para retornar produtos recomendados:
def indicados(segmento, subcategoria, produto):
    # Carregando os dataframes:
    associacoes_nome = segmento.lower().replace(' ', '_') + '_associacoes.csv'
    prop_vendas_nome = segmento.lower().replace(' ', '_') + '_prop_vendas.csv'
    
    associacoes = pd.read_csv(f'./../data/{associacoes_nome}', index_col=0)
    prop_vendas = pd.read_csv(f'./../data/{prop_vendas_nome}', index_col=0)
    
    # Buscando os mais vendidos e os recomendados:
    vendidos = prop_vendas[prop_vendas['Sub-Category'] == subcategoria][
        prop_vendas['Product Name'] != produto]['Product Name'].to_list()

    recomendados = associacoes[associacoes.antecedents == produto]['consequents'].to_list()
    
    # Criando o dataframe e evitando erros para produtos sem recomendações:
    diff = len(vendidos) - len(recomendados)
    if diff > 0:
        [recomendados.append('Sem mais recomendações') for n in range(diff)]
    elif diff < 0:
        [vendidos.append('Sem mais recomendações') for n in range(-diff)]
    else:
        pass
    
    df = pd.DataFrame({'Quem comprou esse produto também comprou': recomendados, 
                   'Você também pode gostar de': vendidos})

    return df.head(5)

# Funções para retornar o dataframe como markdown em uma ui.box:
def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"

def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('-' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])

# Função para criar o gráfico com previsões de vendas:
def prev_vendas(df_prev, seg, subcategoria):
    # Definindo o dataframe com as vendas:
    df_prev = df_prev[(df_prev.Segment==seg) & (df_prev['Sub-Category']==subcategoria)]
    
    # Definindo range de datas:
    idx = pd.date_range(df_prev.index[0], periods=60, freq='M')
    idx = pd.to_datetime(idx.map(lambda x: x.strftime('%Y-%m-01')))

    # Preenchendo os meses sem vendas com zero:
    for n in idx[:48]:
        if n in df_prev.index:
            pass
        else:
            df_prev = pd.concat([df_prev, pd.DataFrame({'Segment': df_prev.Segment.iloc[0], 
                                                        'Sub-Category': df_prev['Sub-Category'].iloc[0], 
                                                        'Sales': 0}, index=[n])])
    df_prev.sort_index(inplace=True)
    
    # Criando o modelo e realizando as previsões:
    model = SARIMAX(endog=df_prev['Sales'], order=(0,1,1), seasonal_order=(0,1,1,12), trend='c', 
                enforce_invertibility=False, enforce_stationarity=False)

    sarimax = model.fit(method='nm')
    
    # Realizando as previsões e criando o dataframe:
    preds = pd.DataFrame([n for n in np.round(sarimax.predict(start=1, end=60))], 
                         index=idx, columns=['Valores'])
    
    preds['Valores'].iloc[:48] = df_prev['Sales'].to_list()[:48]
    
    # Criando o gráfico:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=preds.index[:48], y=preds['Valores'].iloc[:48],
                                 name = 'Observados'))

    fig.add_trace(go.Scatter(x=preds.index[47:], y=preds['Valores'].iloc[47:],
                                 name = 'Previstos'))

    fig.update_layout(title_text=f'Previsão de vendas para a subcategoria {subcategoria} no segmento {seg}', 
                      title_font_color='black', title_x=0.5, title_font_size=10, 
                      legend={'orientation':'h', 
                              'y':1, 'yanchor': 'top', 
                              'x':0, 'xanchor': 'left', 
                              'font':{'size':8, 'color':'black'}, 'itemclick':'toggle'}, 
                      margin=dict(l=0, r=0, b=0, t=15))
    
    fig.update_xaxes(tickfont_size=9, tickfont_color='black')
    fig.update_yaxes(visible=False)
   
    return (pio.to_html(fig, include_plotlyjs='cdn', validate=False))

# Centralizando valores em cartões:
def center_value(value):
    text1 = """<!DOCTYPE html><html><style> 
                    .square {display: flex; align-items: center; 
                            justify-content: center}
                    .square .result{align-self: center; font-size: 18px; 
                                    heigth: 10px; font-family: 'Arial', serif;}
                    </style></head><body><div class="square"><div class="result">"""
    text2 = """</div></div></body></html>"""
        
    return (text1+f'{value}'+text2)
#-------------------------------------------------------------------------------------------------------
#Criando o app:
@app('/painel-vendas')
async def serve(q: Q):
    if not q.client.initialized:
        q.client.initialized=True
        q.client.segmento = 'Consumer'
        q.client.subcategoria = (q.args.subcategoria if q.args.subcategoria in lista_subcat(f'{q.client.segmento}') else lista_subcat(f'{q.client.segmento}')[0])
        q.client.produto = (q.args.produto if q.args.produto in lista_produtos(f'{q.client.segmento}', f'{q.client.subcategoria}') else lista_produtos(f'{q.client.segmento}', f'{q.client.subcategoria}')[0])
        layout(q)
    
    else:
        q.client.segmento = q.args.segmento
        q.client.subcategoria = q.args.subcategoria
        q.client.produto = q.args.produto
        layout(q)


    await q.page.save()
#-------------------------------------------------------------------------------------------------------
def layout(q: Q) -> None:
    q.page['meta'] = ui.meta_card(box='', layouts=[
            ui.layout(breakpoint='xs', zones=[
                ui.zone('main', size='100vh', zones=[
                    ui.zone('header'),
                    ui.zone(name='filtros', direction=ui.ZoneDirection.ROW, zones=[
                        ui.zone('segmento', size='20%'),
                        ui.zone('subcategoria', size='20%'),
                        ui.zone('produto', size='40%'),
                        ui.zone('faturamento', size='20%')
                    ]),
                    ui.zone('content', direction=ui.ZoneDirection.COLUMN, zones=[
                        ui.zone(name='linha1', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('total_vendas', size='20%'),
                            ui.zone('ticket_medio', size='20%'),                        
                            ui.zone('vendas_cat', size='20%'),
                            ui.zone('vendas_cat_ano', size='40%')
                    ]),
                       ui.zone(name='linha2', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('previsoes', size='55%'),
                            ui.zone('recomendacoes', size='45%')
                    ]),
                ]),
                ui.zone('footer')
            ])
        ])
    ])

    q.page['header'] = ui.header_card(
        box = ui.box(
            'header', 
            height='55px'
            ), 
        icon='FinancialSolid',
        icon_color='Yellow',
        title='Análise de vendas com previsão de demanda e recomendações de compras',
        subtitle='',
        )
 
    q.page['segmento'] = ui.form_card(
        box=ui.box('segmento', size='40px'), items=[
                ui.dropdown(name='segmento', label='Escolha o segmento', 
                value=q.client.segmento, choices=[
                    ui.choice(n, n) for n in ['Consumer', 'Corporate', 'Home Office']], 
                        trigger=True, popup='never', required=True)])

    q.page['subcategoria'] = ui.form_card(
        box=ui.box('subcategoria', size='40px'), items=[
                ui.dropdown(name='subcategoria', label='Escolha a subcategoria', 
                value=q.client.subcategoria, choices=[
                    ui.choice(n, n) for n in lista_subcat(f'{q.client.segmento}')], 
                        trigger=True, popup='never', required=True)])

    q.page['produto'] = ui.form_card(
        box=ui.box('produto', size='40px'), items=[
                ui.dropdown(name='produto', label='Escolha o produto', 
                value=q.client.produto, choices=[
                    ui.choice(n, n) for n in lista_produtos(f'{q.client.segmento}', f'{q.client.subcategoria}')], 
                        trigger=True, popup='never', required=True)])

    faturamento_total = str(np.round(df_ticket['total_vendas'].sum()/10e5, 3))+'Mi'

    q.page['faturamento'] = ui.frame_card(
        box=ui.box('faturamento', size='40px'),
        title='Faturamento total no período',
        content=f'{center_value(faturamento_total)}'
    )

    q.page['faturamento'] = ui.frame_card(
        box=ui.box('faturamento', size='40px'),
        title='Faturamento total no período',
        content=f'{center_value(faturamento_total)}'
    )

    # Gerando os gráficos necessários:
    fig1 = total_seg(df_ticket)
    fig2 = ticket_seg(df_ticket)
    fig3 = total_cat(f'{q.client.segmento}', df_cats)
    fig4 = total_cat_ano(f'{q.client.segmento}', df_cats)
    fig5 = prev_vendas(df_prev, f'{q.client.segmento}', f'{q.client.subcategoria}')
    tabela1 = indicados(f'{q.client.segmento}', f'{q.client.subcategoria}', f'{q.client.produto}')

    q.page['total_vendas'] = ui.frame_card(
        box=ui.box('total_vendas', width='100%', height='200px'),
        title='',
        content=fig1)

    q.page['ticket_medio'] = ui.frame_card(
        box=ui.box('ticket_medio', width='100%', height='200px'),
        title='',
        content=fig2)

    q.page['vendas_cat'] = ui.frame_card(
        box=ui.box('vendas_cat', width='100%', height='200px'),
        title='',
        content=fig3)

    q.page['vendas_cat_ano'] = ui.frame_card(
        box=ui.box('vendas_cat_ano', width='100%', height='200px'),
        title='',
        content=fig4)

    q.page['previsoes'] = ui.frame_card(
        box=ui.box('previsoes', width='100%', height='200px'),
        title='',
        content=fig5)

    q.page['recomendacoes'] = ui.form_card(
        box=ui.box('recomendacoes', width='100%', height='200px'), items=[
        ui.text(make_markdown_table(
            fields=tabela1.columns.tolist(),
            rows=tabela1.values.tolist(),
            )),
        ],
    )