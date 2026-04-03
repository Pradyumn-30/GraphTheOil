import pandas as pd
from dotenv import load_dotenv
import os

from langchain_community.graphs import Neo4jGraph

load_dotenv()  # loads variables from .env

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

data = pd.read_csv("../data/oil_imports_23_24_cln.csv")

data = data[['period', 'reporterDesc', 'partnerDesc', 'final_netWgt_MT','primaryValue',
       'final_price_per_mt', 'barrels' ]].drop_duplicates()
print(data.shape)

# Define the Geopolitical Region Mapping based strictly on the given list
region_mapping = {
    # Middle East
    'Bahrain': 'Middle East', 'Iran': 'Middle East', 'Iraq': 'Middle East', 
    'Israel': 'Middle East', 'Kuwait': 'Middle East', 'Oman': 'Middle East', 
    'Qatar': 'Middle East', 'Saudi Arabia': 'Middle East', 'United Arab Emirates': 'Middle East',
    
    # CIS & Central Asia
    'Armenia': 'CIS & Central Asia', 'Azerbaijan': 'CIS & Central Asia', 'Georgia': 'CIS & Central Asia',
    'Kazakhstan': 'CIS & Central Asia', 'Kyrgyzstan': 'CIS & Central Asia', 'Mongolia': 'CIS & Central Asia',
    'Rep. of Moldova': 'CIS & Central Asia', 'Russian Federation': 'CIS & Central Asia',
    'Turkmenistan': 'CIS & Central Asia', 'Ukraine': 'CIS & Central Asia', 'Uzbekistan': 'CIS & Central Asia',
    
    # North America
    'Canada': 'North America', 'Mexico': 'North America', 'USA': 'North America',
    
    # South & Central America & Caribbean
    'Argentina': 'South & Central America & Caribbean', 'Bahamas': 'South & Central America & Caribbean',
    'Belize': 'South & Central America & Caribbean', 'Bermuda': 'South & Central America & Caribbean',
    'Bolivia (Plurinational State of)': 'South & Central America & Caribbean', 
    'Br. Virgin Isds': 'South & Central America & Caribbean', 'Brazil': 'South & Central America & Caribbean', 
    'Chile': 'South & Central America & Caribbean', 'Colombia': 'South & Central America & Caribbean', 
    'Curaçao': 'South & Central America & Caribbean', 'Dominican Rep.': 'South & Central America & Caribbean', 
    'Ecuador': 'South & Central America & Caribbean', 'Guatemala': 'South & Central America & Caribbean', 
    'Guyana': 'South & Central America & Caribbean', 'Nicaragua': 'South & Central America & Caribbean', 
    'Panama': 'South & Central America & Caribbean', 'Paraguay': 'South & Central America & Caribbean', 
    'Peru': 'South & Central America & Caribbean', 'Trinidad and Tobago': 'South & Central America & Caribbean', 
    'Uruguay': 'South & Central America & Caribbean', 'Venezuela': 'South & Central America & Caribbean',
    
    # Europe
    'Albania': 'Europe', 'Austria': 'Europe', 'Belgium': 'Europe', 'Bulgaria': 'Europe', 
    'Croatia': 'Europe', 'Cyprus': 'Europe', 'Czechia': 'Europe', 'Denmark': 'Europe', 
    'Finland': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Greece': 'Europe', 
    'Hungary': 'Europe', 'Ireland': 'Europe', 'Italy': 'Europe', 'Latvia': 'Europe', 
    'Lithuania': 'Europe', 'Malta': 'Europe', 'Netherlands': 'Europe', 'Norway': 'Europe',
    'Other Europe, nes': 'Europe', 'Poland': 'Europe', 'Portugal': 'Europe', 
    'Romania': 'Europe', 'Serbia': 'Europe', 'Slovakia': 'Europe', 'Spain': 'Europe', 
    'Sweden': 'Europe', 'Switzerland': 'Europe', 'Türkiye': 'Europe', 'United Kingdom': 'Europe',
    
    # Africa
    'Algeria': 'Africa', 'Angola': 'Africa', 'Cameroon': 'Africa', 'Chad': 'Africa', 
    'Congo': 'Africa', "Côte d'Ivoire": 'Africa', 'Dem. Rep. of the Congo': 'Africa', 
    'Egypt': 'Africa', 'Equatorial Guinea': 'Africa', 'Gabon': 'Africa', 'Gambia': 'Africa', 
    'Ghana': 'Africa', 'Guinea': 'Africa', 'Liberia': 'Africa', 'Libya': 'Africa', 
    'Mozambique': 'Africa', 'Niger': 'Africa', 'Nigeria': 'Africa', 'Sao Tome and Principe': 'Africa', 
    'Senegal': 'Africa', 'South Africa': 'Africa', 'South Sudan': 'Africa', 'Sudan': 'Africa', 
    'Togo': 'Africa', 'Tunisia': 'Africa', 'Zambia': 'Africa', 'Zimbabwe': 'Africa',
    
    # South Asia
    'India': 'South Asia', 'Pakistan': 'South Asia',

    # South East Asia
    'Brunei Darussalam': 'South East Asia', 'Indonesia': 'South East Asia', 
    "Lao People's Dem. Rep.": 'South East Asia', 'Malaysia': 'South East Asia', 
    'Philippines': 'South East Asia', 'Singapore': 'South East Asia', 
    'Thailand': 'South East Asia', 'Timor-Leste': 'South East Asia', 
    'Viet Nam': 'South East Asia',

    # East Asia & Oceania
    'Australia': 'East Asia & Oceania', 'China': 'East Asia & Oceania', 
    'Japan': 'East Asia & Oceania', 'Marshall Isds': 'East Asia & Oceania', 
    'New Zealand': 'East Asia & Oceania', 'Papua New Guinea': 'East Asia & Oceania', 
    'Rep. of Korea': 'East Asia & Oceania',
    
    # International / Unspecified
    'Areas, nes': 'International / Unspecified', 'Bunkers': 'International / Unspecified'
}

# Map the regions for both the Reporter (Importer) and Partner (Exporter)
# This allows you to query "European imports" or "Middle Eastern exports" easily
data['reporterRegion'] = data['reporterDesc'].map(region_mapping)
data['partnerRegion'] = data['partnerDesc'].map(region_mapping)

# 3. Check for any countries that didn't map (for debugging)
unmapped = data[data['partnerRegion'].isna()]['partnerDesc'].unique()
if len(unmapped) > 0:
    print(f"Warning: The following countries were not in the provided list: {unmapped}")

# 4. Save the new tagged dataset
output_path = 'oil_imports_with_regions.csv'
data.to_csv(output_path, index=False)

print(f"Successfully created {output_path} with regional tagging.")

graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database="78ce8520",
    refresh_schema=True 
)

# 1. Ensure Uniqueness and Indexing
graph.query("CREATE CONSTRAINT country_name_unique IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE;")
graph.query("CREATE CONSTRAINT year_month_id_unique IF NOT EXISTS FOR (t:YearMonth) REQUIRE t.yearMonth IS UNIQUE;")

# Index for region filtering
graph.query("CREATE INDEX country_region_index IF NOT EXISTS FOR (c:Country) ON (c.region);")

# Cypher query to load your preprocessed data
ingest_query = """
UNWIND $data as row
// 1. Ensure all nodes exist uniquely and set the Region property
MERGE (r:Country {name: row.reporterDesc})
ON CREATE SET r.region = row.reporterRegion
ON MATCH SET r.region = row.reporterRegion

MERGE (p:Country {name: row.partnerDesc})
ON CREATE SET p.region = row.partnerRegion
ON MATCH SET p.region = row.partnerRegion

MERGE (t:YearMonth {yearMonth: row.period})

// 2. MERGE the trade relationship
// Using the period in the relationship pattern ensures uniqueness per trade event
MERGE (r)-[f:IMPORTED_FROM {year_month: row.period}]->(p)

// 3. Connect the Importer to the Time node
MERGE (r)-[:IMPORTS_IN]->(t)

// 4. Set the numerical data (Metrics)
SET f.barrels = row.barrels,
    f.value_usd = row.primaryValue,
    f.price_mt = row.final_price_per_mt
"""

# Prepare your data from the cleaned dataframe
data_to_load = data.to_dict('records')

# Execute ingestion
graph.query(ingest_query, params={"data": data_to_load})

graph.refresh_schema()
print(graph.schema)