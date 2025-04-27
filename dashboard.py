import pandas as pd
import numpy as np
import panel as pn
import hvplot.pandas
import os
import requests
from io import StringIO
import param
import holoviews as hv
from bokeh.palettes import Turbo256

pn.extension('tabulator', 'plotly', comms='ipywidgets')

try:
    df = pd.read_csv('https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv')
    if not os.path.exists('GlobalLandTemperaturesByCountry.csv'):
        url = "https://raw.githubusercontent.com/berkeleyearth/data-raw/master/GlobalLandTemperaturesByCountry.csv"
        response = requests.get(url)
        with open('GlobalLandTemperaturesByCountry.csv', 'w') as f:
            f.write(response.text)
    df1 = pd.read_csv('GlobalLandTemperaturesByCountry.csv')
except Exception as e:
    raise Exception("Failed to load datasets. Ensure you have an internet connection and the required files.")

# Additional data for carbon intensity by energy type
carbon_intensity = {
    'energy_source': ['Coal', 'Oil', 'Natural Gas', 'Solar', 'Wind', 'Nuclear', 'Hydroelectric'],
    'carbon_intensity': [820, 720, 490, 48, 12, 12, 24],  # gCO2/kWh
    'color': ['#333333', '#6b4e2b', '#d5b675', '#ffff00', '#a3e4d7', '#5dade2', '#2e86c1']
}
carbon_df = pd.DataFrame(carbon_intensity)

# Data preprocessing
df = df.fillna(0)
df['gdp_per_capita'] = np.where(df['population'] != 0, df['gdp'] / df['population'], 0)
df1['dt'] = pd.to_datetime(df1['dt' if 'dt' in df1.columns else 'Date'])

# Create efficiency metrics
df['carbon_efficiency'] = np.where(df['gdp'] != 0, df['co2'] / df['gdp'] * 1000000, 0)  # CO2 tons per $1M GDP
df['energy_efficiency'] = np.where(df['gdp'] != 0, df['primary_energy_consumption'] / df['gdp'] * 1000, 0)  # kWh per $1k GDP

countries = ['United States', 'China', 'Germany', 'India', 'Japan', 'South Korea']
df_filtered = df1[df1['Country'].isin(countries)].dropna(subset=['AverageTemperature'])

idf = df.interactive()

# Enhanced widgets with more options
year_slider = pn.widgets.IntSlider(name='Year slider', start=1750, end=2020, step=5, value=1900)
yaxis_co2 = pn.widgets.RadioButtonGroup(name='Y axis', options=['co2', 'co2_per_capita', 'carbon_efficiency'], button_type='success')

# Add time range slider for trend analysis
range_slider = pn.widgets.DateRangeSlider(
    name='Time Period',
    start=df1['dt'].min(),
    end=df1['dt'].max(),
    value=(pd.to_datetime('1950-01-01'), df1['dt'].max())
)

continents = ['World', 'Asia', 'Oceania', 'Europe', 'Africa', 'North America', 'South America', 'Antarctica']
co2_pipeline = (
    idf[(idf.year <= year_slider) & (idf.country.isin(continents))]
    .groupby(['country', 'year'])[yaxis_co2].mean()
    .to_frame()
    .reset_index()
    .sort_values(by='year')
    .reset_index(drop=True)
)

co2_plot = co2_pipeline.hvplot(x='year', by='country', y=yaxis_co2, line_width=2, title="CO2 emission by continent")
co2_table = co2_pipeline.pipe(pn.widgets.Tabulator, pagination='remote', page_size=10, sizing_mode='stretch_width')

co2_vs_gdp_scatterplot_pipeline = (
    idf[(idf.year == year_slider) & (~idf.country.isin(continents))]
    .groupby(['country', 'year', 'gdp_per_capita'])['co2'].mean()
    .to_frame()
    .reset_index()
    .sort_values(by='year')
    .reset_index(drop=True)
)

co2_vs_gdp_scatterplot = co2_vs_gdp_scatterplot_pipeline.hvplot(
    x='gdp_per_capita', y='co2', by='country', size=80, kind="scatter", alpha=0.7, legend=False, height=500, width=500
)

yaxis_co2_source = pn.widgets.RadioButtonGroup(name='Y axis', options=['coal_co2', 'oil_co2', 'gas_co2'], button_type='success')
continents_excl_world = ['Asia', 'Oceania', 'Europe', 'Africa', 'North America', 'South America', 'Antarctica']

co2_source_bar_pipeline = (
    idf[(idf.year == year_slider) & (idf.country.isin(continents_excl_world))]
    .groupby(['year', 'country'])[yaxis_co2_source].sum()
    .to_frame()
    .reset_index()
    .sort_values(by='year')
    .reset_index(drop=True)
)

co2_source_bar_plot = co2_source_bar_pipeline.hvplot(
    kind='bar', x='country', y=yaxis_co2_source, title='CO2 source by continent'
)

# NEW: Add carbon intensity comparison chart
carbon_intensity_chart = carbon_df.hvplot.bar(
    x='energy_source', 
    y='carbon_intensity', 
    color='color',
    title='Carbon Intensity by Energy Source',
    xlabel='Energy Source',
    ylabel='Carbon Intensity (gCO2/kWh)',
    height=400
)

df_temp = df1.copy()
selected_countries = ['United States', 'China', 'Germany', 'India', 'Japan', 'South Korea']
df_filtered = df_temp[df_temp['Country'].isin(selected_countries)].dropna(subset=['AverageTemperature'])
df_filtered['Year'] = df_filtered['dt'].dt.year
df_grouped = df_filtered.groupby(['Year', 'Country'])['AverageTemperature'].mean().reset_index()

# Create a dedicated country selector for temperature trends
temp_country_selector = pn.widgets.MultiChoice(
    name='Select Countries',
    options=selected_countries,  # Use all the tech-heavy countries
    value=['United States', 'China', 'Germany', 'India', 'Japan', 'South Korea'],  # Default to all countries
    width=300
)

@pn.depends(temp_country_selector)
def update_plot(countries):
    if not countries:
        return pn.pane.Markdown("No countries selected. Please select at least one country.")
    
    filtered = df_grouped[df_grouped['Country'].isin(countries)]
    if filtered.empty:
        return pn.pane.Markdown("No data available for the selected countries.")
    
    return filtered.hvplot.line(
        x='Year', 
        y='AverageTemperature', 
        by='Country', 
        title='Avg Temperature Trends in Tech-Heavy/Industrial Countries',
        xlabel='Year', 
        ylabel='Temperature (°C)', 
        width=800, 
        height=500, 
        line_width=2
    )

# Define metrics dropdown before using it
metrics_dropdown = pn.widgets.Select(
    name='Select Metric', 
    options=['co2', 'gdp', 'population'], 
    value='co2', 
    width=200
)

# Define efficiency dropdown before using it
efficiency_dropdown = pn.widgets.Select(
    name='Select Efficiency Metric', 
    options=['carbon_efficiency', 'energy_efficiency'], 
    value='carbon_efficiency', 
    width=200
)

# Create a separate country selector for the metrics section
metrics_country_selector = pn.widgets.MultiChoice(
    name='Select Countries',
    options=df['country'].unique().tolist(),
    value=['United States'],  # Default selection
    width=300
)

# Update the dependent functions to use the metrics_country_selector
@pn.depends(metrics_country_selector, metrics_dropdown)
def get_country_metric_plot(selected_countries, metric):
    country_data = df[df['country'].isin(selected_countries)].sort_values('year')
    if country_data.empty:
        return pn.pane.Markdown("No data available for the selected countries.")
    
    return country_data.hvplot.line(
        x='year',
        y=metric,
        by='country',
        title=f'{metric.capitalize()} Trends for Selected Countries',
        xlabel='Year',
        ylabel=metric.capitalize(),
        line_width=2,
        height=400,
        width=800,
        hover_cols=['country', 'year']
    )

@pn.depends(metrics_country_selector, efficiency_dropdown)
def get_country_efficiency_plot(selected_countries, efficiency_metric):
    country_data = df[df['country'].isin(selected_countries)].sort_values('year')
    if country_data.empty:
        return pn.pane.Markdown("No data available for the selected countries.")
    
    return country_data.hvplot.line(
        x='year',
        y=efficiency_metric,
        by='country',
        title=f'{efficiency_metric.replace("_", " ").capitalize()} for Selected Countries',
        xlabel='Year',
        ylabel=efficiency_metric.replace("_", " ").capitalize(),
        line_width=2,
        height=400,
        width=800,
        hover_cols=['country', 'year']
    )

# Define temperature_text before using it
temperature_text = pn.pane.Markdown("""
### The Climate Cost of Innovation

#### Temperature trends in tech-centric nations as automation & AI advance

Select countries from the dropdown to compare how average temperatures have changed over time in nations that have been leaders in technological development and industrial growth.

This visualization helps us understand if there's a correlation between technological advancement and climate impact over the decades.
""")

# Define tech_adoption_df before using it
tech_adoption_data = {
    'country': ['United States', 'China', 'Germany', 'Japan', 'South Korea', 'India', 'United Kingdom', 'Canada', 'France', 'Australia'],
    'tech_economy_pct_2020': [22.1, 17.8, 15.3, 12.5, 21.6, 8.2, 16.7, 14.9, 13.8, 11.2]
}
tech_adoption_df = pd.DataFrame(tech_adoption_data)

# Define get_tech_vs_emissions before using it
@pn.depends(year_slider)
def get_tech_vs_emissions(year):
    tech_emissions = df[(df.year == year) & (df.country.isin(tech_adoption_df['country']))].merge(tech_adoption_df, on='country')
    tech_correlation = tech_emissions['tech_economy_pct_2020'].corr(tech_emissions['co2_per_capita'])
    plot = tech_emissions.hvplot.scatter(
        x='tech_economy_pct_2020', y='co2_per_capita', by='country', title=f'Tech Economy vs CO2 Emissions (Year: {year})',
        xlabel='Digital/Tech Economy (% of GDP)', ylabel='CO2 per Capita', height=400, width=600, size=100,
        hover_cols=['gdp_per_capita', 'population'], alpha=0.7
    )
    correlation_text = pn.pane.Markdown(f"""
    ### Correlation Analysis: Tech Economy vs Carbon Emissions

    Correlation coefficient: **{tech_correlation:.3f}**

    {'Strong positive correlation' if tech_correlation > 0.7 else 
     'Moderate positive correlation' if tech_correlation > 0.3 else
     'Weak positive correlation' if tech_correlation > 0 else
     'Strong negative correlation' if tech_correlation < -0.7 else
     'Moderate negative correlation' if tech_correlation < -0.3 else
     'Weak negative correlation'}

    This suggests that {'as technology adoption increases, carbon emissions tend to increase as well' if tech_correlation > 0 else 'technology adoption may help reduce carbon emissions'}.
    """)
    return pn.Column(plot, correlation_text)

# Updated Country Metrics Tab
country_analysis_tab = pn.Tabs(
    ('Country Metrics', pn.Column(
        pn.Row(
            pn.pane.Markdown("## Country-Level Analysis"),
            metrics_country_selector
        ),
        pn.Row(
            pn.Column(metrics_dropdown, get_country_metric_plot),
            pn.Column(efficiency_dropdown, get_country_efficiency_plot)
        )
    )),
    ('Temperature Trends', pn.Row(
        pn.Column(
            pn.pane.Markdown("## The Flip Side of Progress: Temperature Trends in Tech-Centric Nations"),
            pn.Row(
                pn.Column(temperature_text, temp_country_selector, width=300),
                update_plot
            )
        )
    ))
)

future_years = np.arange(2020, 2050, 5)
future_emissions = {
    'Year': future_years,
    'Business as Usual': [36.8, 38.2, 39.7, 41.2, 42.8, 44.5],
    'Green Tech Scenario': [36.8, 35.3, 33.0, 30.1, 26.5, 22.2],
    'AI Optimized Energy': [36.8, 34.0, 30.5, 27.8, 22.1, 17.6] # NEW: Added scenario
}
future_df = pd.DataFrame(future_emissions)

# NEW: Enhanced future plot with an additional scenario
future_plot = pn.pane.HoloViews(
    future_df.hvplot.line(
        x='Year', 
        y=['Business as Usual', 'Green Tech Scenario', 'AI Optimized Energy'], 
        title='AI & Green Tech: Potential Future Emission Pathways',
        xlabel='Year', 
        ylabel='Global CO2 Emissions (Gt)', 
        height=400, 
        width=600, 
        line_width=3,
        value_label='Emissions (Gt CO2)'
    )
)

# NEW: Interactive scenario builder
class ScenarioBuilder(param.Parameterized):
    ai_adoption_rate = param.Number(0.5, bounds=(0.0, 1.0), step=0.1)
    renewable_adoption_rate = param.Number(0.3, bounds=(0.0, 1.0), step=0.1)
    efficiency_improvement = param.Number(0.2, bounds=(0.0, 1.0), step=0.1)
    
    @param.depends('ai_adoption_rate', 'renewable_adoption_rate', 'efficiency_improvement', watch=True)
    def update_scenario(self):
        self.scenario_plot.object = self.build_scenario()
    
    def build_scenario(self):
        base = future_df['Business as Usual'].values
        green = future_df['Green Tech Scenario'].values
        
        # Calculate custom scenario based on parameters
        custom_reductions = (self.ai_adoption_rate * 0.3 + 
                            self.renewable_adoption_rate * 0.5 + 
                            self.efficiency_improvement * 0.2)
        
        scenario = base * (1 - custom_reductions * np.linspace(0, 1, len(base)))
        
        scenario_df = pd.DataFrame({
            'Year': future_years,
            'Business as Usual': base,
            'Your Custom Scenario': scenario
        })
        
        return scenario_df.hvplot.line(
            x='Year',
            y=['Business as Usual', 'Your Custom Scenario'],
            title='Your Custom Emissions Scenario',
            xlabel='Year',
            ylabel='Global CO2 Emissions (Gt)',
            height=300,
            width=500,
            line_width=2
        )
    
    def __init__(self, **params):
        super().__init__(**params)
        self.scenario_plot = pn.pane.HoloViews(self.build_scenario(), height=350)
    
    def view(self):
        return pn.Column(
            pn.Param(
                self.param, 
                widgets={
                    'ai_adoption_rate': pn.widgets.FloatSlider,
                    'renewable_adoption_rate': pn.widgets.FloatSlider,
                    'efficiency_improvement': pn.widgets.FloatSlider
                }
            ),
            self.scenario_plot
        )

scenario_builder = ScenarioBuilder()

emission_reduction_text = pn.pane.Markdown("""
### AI's Dual Impact on Climate

**Technology Innovation Paradox:**
- AI and automation typically increase economic growth
- But increased production often leads to higher emissions
- On the other hand, AI enables energy optimization and efficiency

**Green Tech Potential:**
- AI-powered smart grids could reduce energy waste by 15-30%
- Algorithmic optimization may reduce transportation emissions by 10-20%
- AI-enabled material science could create more efficient energy solutions

The trajectory we take depends on policy choices and how we direct technological innovation.
""")

try:
    climate_impact_image = pn.pane.PNG('climate_impact.png', sizing_mode='scale_width')
except Exception:
    climate_impact_image = pn.pane.Markdown("## Climate Impact Visualization\n\n(Image not found. Please place climate_impact.png in the same folder as this script)")

try:
    climate_day_image = pn.pane.PNG('climate_day.png', sizing_mode='scale_width')
except Exception:
    climate_day_image = pn.pane.Markdown("## 🌍 Climate Dashboard\n\n(Image not found. Please place climate_day.png in the same folder as this script)")

# NEW: Add climate policy effectiveness visualization
policy_data = {
    'Policy Type': ['Carbon Tax', 'Renewable Subsidies', 'Efficiency Standards', 'R&D Investment', 'Trade Agreements'],
    'Cost Effectiveness': [8.7, 7.2, 6.5, 9.2, 4.8],
    'Emission Reduction': [7.9, 8.3, 5.6, 6.4, 3.2],
    'Political Feasibility': [4.3, 7.8, 6.9, 8.5, 7.1],
    'Implementation Time': [8.6, 6.5, 5.3, 3.2, 4.1]
}
policy_df = pd.DataFrame(policy_data)

# Transform for radar chart
import plotly.graph_objects as go
from math import pi

def create_radar_chart():
    categories = policy_df.columns[1:].tolist()
    fig = go.Figure()
    
    for i, policy in enumerate(policy_df['Policy Type']):
        values = policy_df.iloc[i, 1:].tolist()
        values.append(values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=policy
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title="Climate Policy Effectiveness Comparison",
        showlegend=True,
        height=450
    )
    
    return fig

policy_radar = pn.pane.Plotly(create_radar_chart())

# TABS for different sections
co2_trends_tab = pn.Tabs(
    ('CO2 Emissions', pn.Column(
        pn.Row(
            pn.Column(
                pn.pane.Markdown("## CO2 Trajectories: Tech Innovation vs Climate Impact"),
                yaxis_co2, 
                co2_plot.panel(width=700), 
                margin=(0,25)
            ), 
            co2_table.panel(width=500)
        )
    )),
    ('Economy & Carbon', pn.Row(
        pn.Column(
            pn.pane.Markdown("## Innovation's Price: Economic Growth vs Environmental Cost"),
            co2_vs_gdp_scatterplot.panel(width=600), 
            margin=(0,25)
        )
    )),
    ('Carbon Sources', pn.Row(
        pn.Column(
            pn.pane.Markdown("## Sources of Innovation's Carbon Footprint"),
            yaxis_co2_source, 
            co2_source_bar_plot.panel(width=600)
        ),
        pn.Column(
            pn.pane.Markdown("## Carbon Intensity of Energy Sources"),
            pn.pane.HoloViews(carbon_intensity_chart, width=600)
        )
    ))
)

tech_impact_tab = pn.Tabs(
    ('Tech vs Emissions', pn.Row(
        pn.Column(
            pn.pane.Markdown("## Digital Economy vs Carbon Footprint"),
            get_tech_vs_emissions,
            margin=(0,25),
            width=800
        ),
        pn.Column(
            pn.pane.Markdown("## Climate Impact Visualization"),
            climate_impact_image,
            margin=(0,25),
            width=600
        )
    )),
    ('Future Scenarios', pn.Row(
        pn.Column(
            pn.pane.Markdown("## AI's Potential Future Impact on Global Emissions"),
            future_plot,
            margin=(0,25)
        ),
        pn.Column(
            emission_reduction_text,
            margin=(0,25)
        )
    )),
    ('Create Your Scenario', pn.Column(
        pn.pane.Markdown("## Build Your Own Emissions Scenario"),
        pn.pane.Markdown("""
        Use the sliders below to adjust parameters and see how different levels of technology adoption and policy implementations
        could affect future emissions. This tool helps visualize the potential impact of your own strategies.
        """),
        scenario_builder.view()
    )),
    ('Policy Analysis', pn.Column(
        pn.pane.Markdown("## Climate Policy Effectiveness"),
        pn.pane.Markdown("""
        This visualization compares different climate policies across multiple dimensions:
        - **Cost Effectiveness**: Economic efficiency of the policy
        - **Emission Reduction**: Direct impact on reducing emissions
        - **Political Feasibility**: How realistic implementation is
        - **Implementation Time**: How quickly benefits can be realized
        """),
        policy_radar
    ))
)

template = pn.template.FastListTemplate(
    title='Innovation\'s Carbon Footprint: The Climate Impact of Tech Growth', 
    theme='dark',
    sidebar=[
        pn.pane.Markdown("# AI, Automation & Climate Change"), 
        pn.pane.Markdown("#### Even as we move toward AI and automation, carbon footprints remain.\n\nThis dashboard examines how industrial growth and tech-centric countries impact emissions.\n\nWhat does the 'flip side' of innovation look like through climate data?", width=300), 
        climate_day_image,
        pn.pane.Markdown("## Time Controls"),   
        year_slider,
        range_slider
    ],
    main=[
        pn.Row(pn.pane.Markdown("## Explore the relationship between technology, innovation and climate change through these interactive visualizations")),
        co2_trends_tab,
        country_analysis_tab,
        tech_impact_tab
    ],
    accent_base_color="#0C5DA5",
    header_background="#0C5DA5",
)
    template.servable()
