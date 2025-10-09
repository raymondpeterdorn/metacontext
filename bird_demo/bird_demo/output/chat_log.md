raymondpeterdorn: 🔍 COLUMN FORENSICS DETECTIVE - Individual Column Investigation

You are a data forensics specialist conducting deep investigation into individual dataset columns.
Your specialty is uncovering the hidden stories behind poorly named, cryptic, or suspicious columns.


🕵️ CONDUCT FORENSIC COLUMN-BY-COLUMN INVESTIGATION:

📊 EVIDENCE TO ANALYZE:
- Dataset: csv.csv
- Project Context: No project context available
- Codebase Context: No code context

🧠 SEMANTIC KNOWLEDGE FROM CODEBASE:
No semantic knowledge available.

📋 COLUMN DATA TO INVESTIGATE:
{'species_name': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': ['Haemorhous mexicanus', 'Pica pica', 'Pica pica']}, 'taxonomic_family': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': ['Fringillidae', 'Corvidae', 'Corvidae']}, 'taxonomic_order': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 4, 'sample_values': ['Passeriformes', 'Passeriformes', 'Passeriformes']}, 'asdawas': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [8.91156072292315, 18.831008742960407, 21.63353417277567]}, 'beak_length': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [1.2084595336364423, 3.464735162654236, 3.5949562606268004]}, 'weight_g': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [22.63545835787437, 221.9276595225436, 230.1582173720443]}, 'nocturnal_diurnal': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 2, 'sample_values': ['Diurnal', 'Diurnal', 'Diurnal']}, 'brrrrkk': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 158, 'sample_values': [datetime.date(2025, 7, 20), datetime.date(2024, 12, 20), datetime.date(2025, 1, 3)]}, 'diet_types': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': [{'seeds': 1.0}, {'insects': 0.5, 'carrion': 0.3, 'seeds': 0.2}, {'insects': 0.5, 'carrion': 0.3, 'seeds': 0.2}]}, 'closest_relatives': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': [['Haemorhous purpureus', 'Haemorhous cassinii'], ['Pica hudsonia'], ['Pica hudsonia']]}, 'latitude': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [32.346401128781864, 38.44235705433985, 43.831044771135495]}, 'longitude': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [-91.0332914317774, -123.20837942455418, -90.37110890379113]}, 'altitude_m': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [557.0704794605397, 716.3898689319495, 206.67436170278563]}, 'habitat_type': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 4, 'sample_values': ['Urban', 'Grassland', 'Grassland']}, 'weather_condition': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': ['Cloudy', 'Windy', 'Clear']}, 'temperature_c': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [19.23505378184511, 26.60126383416036, 20.507738467172867]}, 'observation_time': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 184, 'sample_values': ['07:55', '15:12', '11:50']}, 'diet_dict': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': [{'seeds': 1.0}, {'insects': 0.5, 'carrion': 0.3, 'seeds': 0.2}, {'insects': 0.5, 'carrion': 0.3, 'seeds': 0.2}]}, 'primary_diet': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 4, 'sample_values': ['seeds', 'insects', 'insects']}, 'is_nocturnal': {'dtype': 'bool', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 2, 'sample_values': [False, False, False]}, 'blwl': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [0.05338789763080041, 0.015612002443085672, 0.015619499932151691]}}

🔍 FORENSIC INVESTIGATION - For EACH column:

1. **NAME ANALYSIS**: Is this column name vague, cryptic, or misleading?
2. **SEMANTIC KNOWLEDGE**: Check if we have extracted definitions from Pydantic models or code comments
3. **HIDDEN MEANING**: Based on codebase context, what business logic produces these values?
4. **CROSS-REFERENCE**: Map suspicious names to potential code locations.
5. **PATTERN DETECTION**: Are there magic numbers, unusual ranges, or suspicious patterns?

🎯 EFFICIENCY REQUIREMENTS:
- Maximum response: 1200 characters total
- Use precise technical terminology
- Focus on highest-value insights

🔍 REQUIRED ANALYSIS FORMAT:
1. **DOMAIN ANALYSIS**: Overall business domain and context of this dataset
2. **DATA QUALITY**: Overall structural quality assessment
3. **COLUMN INTERPRETATIONS**: For each column, provide nested analysis
4. **BUSINESS VALUE**: How this data provides business insights

⚠️ CRITICAL Output Format Rules:
- Output must be **valid JSON only** matching DataAIEnrichment schema
- column_interpretations must contain nested ai_enrichment objects
- relationship_to_other_columns: Must be a list ["relation1", "relation2"] or null
- Never include markdown or prose
- Focus on business logic and data relationships

📋 Required Output Structure (DataAIEnrichment format):
{
  "domain_analysis": "Overall domain this dataset represents and business context",
  "data_quality_assessment": "Overall assessment of data structure and quality",
  "column_interpretations": {
    "column_name": {
      "ai_enrichment": {
        "semantic_meaning": "What this column represents",
        "domain_context": "Business domain context", 
        "usage_guidance": "How to use this column",
        "data_quality_assessment": "Quality notes",
        "relationship_to_other_columns": ["related_col1", "related_col2"]
      }
    }
  },
  "business_value_assessment": "How this data provides business value and insights"
}

📋 Return optimized JSON in DataAIEnrichment format with proper nested structure.


Context Information:
file_name: csv.csv
project_summary: No project context available
code_summary: No code context

columns_data:
altitude_m:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 557.0704794605397
  - 716.3898689319495
  - 206.67436170278563
  unique_count: 200
asdawas:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 8.91156072292315
  - 18.831008742960407
  - 21.63353417277567
  unique_count: 200
beak_length:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 1.2084595336364423
  - 3.464735162654236
  - 3.5949562606268004
  unique_count: 200
blwl:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 0.05338789763080041
  - 0.015612002443085672
  - 0.015619499932151691
  unique_count: 200
brrrrkk:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 2025-07-20
  - 2024-12-20
  - 2025-01-03
  unique_count: 158
closest_relatives:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - - Haemorhous purpureus
    - Haemorhous cassinii
  - - Pica hudsonia
  - - Pica hudsonia
  unique_count: 5
diet_dict:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - seeds: 1.0
  - carrion: 0.3
    insects: 0.5
    seeds: 0.2
  - carrion: 0.3
    insects: 0.5
    seeds: 0.2
  unique_count: 5
diet_types:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - seeds: 1.0
  - carrion: 0.3
    insects: 0.5
    seeds: 0.2
  - carrion: 0.3
    insects: 0.5
    seeds: 0.2
  unique_count: 5
habitat_type:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Urban
  - Grassland
  - Grassland
  unique_count: 4
is_nocturnal:
  dtype: bool
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - false
  - false
  - false
  unique_count: 2
latitude:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 32.346401128781864
  - 38.44235705433985
  - 43.831044771135495
  unique_count: 200
longitude:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - -91.0332914317774
  - -123.20837942455418
  - -90.37110890379113
  unique_count: 200
nocturnal_diurnal:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Diurnal
  - Diurnal
  - Diurnal
  unique_count: 2
observation_time:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 07:55
  - '15:12'
  - '11:50'
  unique_count: 184
primary_diet:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - seeds
  - insects
  - insects
  unique_count: 4
species_name:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Haemorhous mexicanus
  - Pica pica
  - Pica pica
  unique_count: 5
taxonomic_family:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Fringillidae
  - Corvidae
  - Corvidae
  unique_count: 5
taxonomic_order:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Passeriformes
  - Passeriformes
  - Passeriformes
  unique_count: 4
temperature_c:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 19.23505378184511
  - 26.60126383416036
  - 20.507738467172867
  unique_count: 200
weather_condition:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Cloudy
  - Windy
  - Clear
  unique_count: 5
weight_g:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 22.63545835787437
  - 221.9276595225436
  - 230.1582173720443
  unique_count: 200

semantic_column_knowledge: No semantic knowledge available.

GitHub Copilot: {"domain_analysis":"ornithological field observations; species, morphometrics, location, environment","data_quality_assessment":"no missing values; mixed types; some cryptic names (asdawas,brrrrkk,blwl); nested structures present","column_interpretations":{"species_name":{"ai_enrichment":{"semantic_meaning":"binomial","domain_context":"orn","usage_guidance":"identifier/grouping","data_quality_assessment":"complete,5 uniques","relationship_to_other_columns":["taxonomic_family"]}},"taxonomic_family":{"ai_enrichment":{"semantic_meaning":"family name","domain_context":"orn","usage_guidance":"taxonomy grouping","data_quality_assessment":"complete","relationship_to_other_columns":["species_name"]}},"taxonomic_order":{"ai_enrichment":{"semantic_meaning":"order name","domain_context":"orn","usage_guidance":"higher taxonomy","data_quality_assessment":"complete","relationship_to_other_columns":["taxonomic_family"]}},"asdawas":{"ai_enrichment":{"semantic_meaning":"unknown numeric index","domain_context":"morphometrics/score","usage_guidance":"treat as continuous feature; investigate origin","data_quality_assessment":"complete,200 unique,cryptic","relationship_to_other_columns":["beak_length"]}},"beak_length":{"ai_enrichment":{"semantic_meaning":"beak length (units?)","domain_context":"morphology","usage_guidance":"size comparisons,allometry","data_quality_assessment":"complete,continuous","relationship_to_other_columns":["weight_g"]}},"weight_g":{"ai_enrichment":{"semantic_meaning":"mass grams","domain_context":"morphology/ecology","usage_guidance":"biomass,condition metrics","data_quality_assessment":"complete,continuous","relationship_to_other_columns":["beak_length"]}},"nocturnal_diurnal":{"ai_enrichment":{"semantic_meaning":"activity period label","domain_context":"behavior","usage_guidance":"filter/time-based analysis","data_quality_assessment":"binary,consistent","relationship_to_other_columns":["is_nocturnal"]}},"brrrrkk":{"ai_enrichment":{"semantic_meaning":"date (likely observation)","domain_context":"temporal","usage_guidance":"use as observation_date","data_quality_assessment":"dates parsed,158 uniques","relationship_to_other_columns":["observation_time"]}},"diet_types":{"ai_enrichment":{"semantic_meaning":"diet composition dict","domain_context":"ecology","usage_guidance":"aggregate to primary diet or proportions","data_quality_assessment":"nested dicts,consistent","relationship_to_other_columns":["primary_diet"]}},"closest_relatives":{"ai_enrichment":{"semantic_meaning":"nearby taxa list","domain_context":"phylogeny","usage_guidance":"similarity/phylo checks","data_quality_assessment":"lists,low cardinality","relationship_to_other_columns":["species_name"]}},"latitude":{"ai_enrichment":{"semantic_meaning":"decimal lat","domain_context":"geography","usage_guidance":"map,spatial joins","data_quality_assessment":"valid floats","relationship_to_other_columns":["longitude"]}},"longitude":{"ai_enrichment":{"semantic_meaning":"decimal lon","domain_context":"geography","usage_guidance":"map,spatial joins","data_quality_assessment":"valid floats","relationship_to_other_columns":["latitude"]}},"altitude_m":{"ai_enrichment":{"semantic_meaning":"elevation meters","domain_context":"environment","usage_guidance":"environmental covariate","data_quality_assessment":"continuous,complete","relationship_to_other_columns":["temperature_c"]}},"habitat_type":{"ai_enrichment":{"semantic_meaning":"habitat label","domain_context":"ecology","usage_guidance":"group/filter habitat-specific analyses","data_quality_assessment":"categorical,4 values","relationship_to_other_columns":["species_name"]}},"weather_condition":{"ai_enrichment":{"semantic_meaning":"short weather label","domain_context":"environment","usage_guidance":"covariate for behavior","data_quality_assessment":"categorical","relationship_to_other_columns":["temperature_c"]}},"temperature_c":{"ai_enrichment":{"semantic_meaning":"air temperature °C","domain_context":"environment","usage_guidance":"covariate","data_quality_assessment":"continuous,complete","relationship_to_other_columns":["weather_condition"]}},"observation_time":{"ai_enrichment":{"semantic_meaning":"time of day HH:MM","domain_context":"temporal","usage_guidance":"circadian analyses","data_quality_assessment":"strings,184 uniques","relationship_to_other_columns":["brrrrkk"]}},"diet_dict":{"ai_enrichment":{"semantic_meaning":"diet composition dict","domain_context":"ecology","usage_guidance":"use like diet_types; normalize","data_quality_assessment":"redundant with diet_types","relationship_to_other_columns":["diet_types"]}},"primary_diet":{"ai_enrichment":{"semantic_meaning":"dominant diet category","domain_context":"ecology","usage_guidance":"categorical predictor","data_quality_assessment":"consistent with diet_dict","relationship_to_other_columns":["diet_dict"]}},"is_nocturnal":{"ai_enrichment":{"semantic_meaning":"boolean nocturnality","domain_context":"behavior","usage_guidance":"binary filter","data_quality_assessment":"boolean,consistent","relationship_to_other_columns":["nocturnal_diurnal"]}},"blwl":{"ai_enrichment":{"semantic_meaning":"unknown ratio/index","domain_context":"morphometrics","usage_guidance":"treat as continuous; investigate formula","data_quality_assessment":"complete,cryptic","relationship_to_other_columns":["beak_length"]}}},"business_value_assessment":"Enables species-level ecological analyses, spatial distribution mapping, trait-based models; cryptic fields require provenance to maximize reliability."}

raymondpeterdorn: 🧠 ROLE: Data Forensics Analyst

You are an expert data analyst investigating tabular datasets to uncover data meanings,
quality patterns, and business logic from structural evidence.
Think like a **data archaeologist**:
- Infer dataset purpose from structure and column patterns
- Detect business rules and data validation logic from constraints
- Identify data quality issues and transformation requirements  
- Prioritize precision, brevity, and factual reasoning

Your output will populate a predefined YAML schema describing the dataset analysis.


🕵️ TASK: Conduct targeted dataset analysis based on provided evidence.

📊 Dataset Context:
- Dataset: csv.csv
- Structure: 200 rows x 21 columns
- Project: No project context available

⚙️ Schema Reference (summary only):
- ai_interpretation: any
- ai_confidence: any
- ai_domain_context: any
- usage_guidance: any
- hidden_meaning: any  # Explanations, business logic, or context buried in code comments, variable names...
- suspicious_patterns: any  # List of oddities found: poorly named variables/columns, magic numbers, confusing...
- cross_references: any  # Mapping of confusing field/variable names to where in the codebase they are crea...
- detective_insights: any  # Reverse-engineered understanding of developer intent, business requirements, or ...
- domain_analysis: any  # The domain this data represents and its key characteristics in a business contex...
- data_quality_assessment: any  # Assessment of how well-structured and clear the overall schema is.
- column_interpretations: any  # Detailed interpretations of each column in the dataset.
- business_value_assessment: any  # How this data might provide business value and insights.

🎯 Analysis Guidelines:
1. Use only verifiable evidence from dataset structure and column information.
2. Infer business meaning and data quality — do not restate technical details verbatim.
3. When information is missing, leave the corresponding field as an empty string.
4. Use concise, factual descriptions suitable for data governance metadata.
5. Avoid verbose speculation, repetition, or narrative commentary.

📏 Efficiency Rules:
- Total output ≤ 1200 characters
- Each field ≤ 200 characters
- Avoid redundant phrasing or filler words

⚠️ CRITICAL Output Format Rules:
- ai_confidence: Must be exactly "LOW", "MEDIUM", or "HIGH" (not descriptive text)
- suspicious_patterns: Must be a list ["pattern1", "pattern2"] or null
- cross_references: Must be a dict {"field": "explanation"} or null  
- column_interpretations: Must be a dict with column names as keys, each containing:
  {"semantic_meaning": "...", "domain_context": "...", "usage_guidance": "...", "data_quality_assessment": "..."}
- Never include markdown or prose — output must be **valid JSON only**
- Use technical language to explain *why* columns exist, not *what* they contain
- Focus on business logic and data relationships
- Skip commentary about the task or user

� Example Output Structure:
{
  "ai_interpretation": "Brief dataset purpose",
  "ai_confidence": "HIGH",
  "ai_domain_context": "Domain area",
  "usage_guidance": "How to use this data", 
  "hidden_meaning": "Hidden business logic",
  "suspicious_patterns": ["pattern1", "pattern2"],
  "cross_references": {"field": "explanation"},
  "detective_insights": "Key insights",
  "domain_analysis": "Domain analysis", 
  "data_quality_assessment": "Quality assessment",
  "column_interpretations": {
    "column_name": {
      "semantic_meaning": "What this column represents",
      "domain_context": "Business domain context",
      "usage_guidance": "How to use this column",
      "data_quality_assessment": "Quality notes"
    }
  },
  "business_value_assessment": "Business value"
}

�💡 Output Format:
Return structured JSON matching the injected schema class.

Return valid JSON only.


Context Information:
file_name: csv.csv
rows: 200
num_columns: 21
project_summary: No project context available

columns:
- species_name
- taxonomic_family
- taxonomic_order
- asdawas
- beak_length
- weight_g
- nocturnal_diurnal
- brrrrkk
- diet_types
- closest_relatives
- latitude
- longitude
- altitude_m
- habitat_type
- weather_condition
- temperature_c
- observation_time
- diet_dict
- primary_diet
- is_nocturnal
- blwl


GitHub Copilot: {
  "ai_interpretation":"Field-scale ornithological observations for trait, spatial and behavioral analysis",
  "ai_confidence":"MEDIUM",
  "ai_domain_context":"Ornithology / field ecology",
  "usage_guidance":"Use for trait modeling, spatial mapping, and behavior covariates; resolve cryptic fields before publishing",
  "hidden_meaning":"Cryptic numeric fields likely derived indices; diet stored as both raw composition and aggregated label; date/time split",
  "suspicious_patterns":["cryptic names (asdawas,brrrrkk,blwl)","redundant fields (diet_dict,diet_types; is_nocturnal,nocturnal_diurnal)","nested structures (dicts,lists)"],
  "cross_references":null,
  "detective_insights":"Export mixes raw and derived features from processing pipeline; temporal data split; diet normalized and aggregated",
  "domain_analysis":"Species observations with taxonomy, morphometrics, geolocation, environment, diet and timestamps",
  "data_quality_assessment":"Structurally complete, no nulls; nested values and cryptic column names risk misinterpretation",
  "column_interpretations":{
    "species_name":{"semantic_meaning":"Latin binomial identifier","domain_context":"taxonomy","usage_guidance":"primary key for grouping","data_quality_assessment":"complete,low cardinality"},
    "taxonomic_family":{"semantic_meaning":"family-level taxon","domain_context":"taxonomy","usage_guidance":"aggregate analyses","data_quality_assessment":"complete"},
    "taxonomic_order":{"semantic_meaning":"order-level taxon","domain_context":"taxonomy","usage_guidance":"higher-level grouping","data_quality_assessment":"complete"},
    "asdawas":{"semantic_meaning":"unknown numeric index/score","domain_context":"derived morphometric/score","usage_guidance":"investigate provenance before use","data_quality_assessment":"numeric,cryptic"},
    "beak_length":{"semantic_meaning":"beak length (unit unspecified)","domain_context":"morphology","usage_guidance":"size/allometry analyses","data_quality_assessment":"continuous"},
    "weight_g":{"semantic_meaning":"mass in grams","domain_context":"morphology","usage_guidance":"condition and biomass metrics","data_quality_assessment":"continuous"},
    "nocturnal_diurnal":{"semantic_meaning":"activity label","domain_context":"behavior","usage_guidance":"categorical filter","data_quality_assessment":"redundant with is_nocturnal"},
    "brrrrkk":{"semantic_meaning":"date (observation)","domain_context":"temporal","usage_guidance":"use as observation_date","data_quality_assessment":"parsed dates"},
    "diet_types":{"semantic_meaning":"diet composition dict","domain_context":"ecology","usage_guidance":"normalize proportions","data_quality_assessment":"nested dicts"},
    "closest_relatives":{"semantic_meaning":"list of similar taxa","domain_context":"phylogeny","usage_guidance":"phylogenetic checks","data_quality_assessment":"list,low cardinality"},
    "latitude":{"semantic_meaning":"decimal latitude","domain_context":"geography","usage_guidance":"spatial joins/mapping","data_quality_assessment":"valid floats"},
    "longitude":{"semantic_meaning":"decimal longitude","domain_context":"geography","usage_guidance":"spatial joins/mapping","data_quality_assessment":"valid floats"},
    "altitude_m":{"semantic_meaning":"elevation meters","domain_context":"environment","usage_guidance":"environmental covariate","data_quality_assessment":"continuous"},
    "habitat_type":{"semantic_meaning":"habitat label","domain_context":"ecology","usage_guidance":"stratify analyses","data_quality_assessment":"categorical"},
    "weather_condition":{"semantic_meaning":"short weather label","domain_context":"environment","usage_guidance":"covariate for behavior","data_quality_assessment":"categorical"},
    "temperature_c":{"semantic_meaning":"air temperature °C","domain_context":"environment","usage_guidance":"covariate","data_quality_assessment":"continuous"},
    "observation_time":{"semantic_meaning":"time of day HH:MM","domain_context":"temporal","usage_guidance":"combine with date for timestamp","data_quality_assessment":"string,partial granularity"},
    "diet_dict":{"semantic_meaning":"diet composition dict (duplicate)","domain_context":"ecology","usage_guidance":"prefer normalized form","data_quality_assessment":"redundant"},
    "primary_diet":{"semantic_meaning":"dominant diet category","domain_context":"ecology","usage_guidance":"categorical predictor","data_quality_assessment":"consistent with diet_dict"},
    "is_nocturnal":{"semantic_meaning":"boolean nocturnality","domain_context":"behavior","usage_guidance":"binary filter","data_quality_assessment":"consistent"},
    "blwl":{"semantic_meaning":"unknown ratio/index (cryptic)","domain_context":"derived morphometric","usage_guidance":"investigate formula before modeling","data_quality_assessment":"numeric,cryptic"}
  },
  "business_value_assessment":"Enables trait-based ecological modeling, species distribution and behavior studies; provenance of derived fields must be clarified for reproducible insights"
}

raymondpeterdorn: 🔍 COLUMN FORENSICS DETECTIVE - Individual Column Investigation

You are a data forensics specialist conducting deep investigation into individual dataset columns.
Your specialty is uncovering the hidden stories behind poorly named, cryptic, or suspicious columns.


🕵️ CONDUCT FORENSIC COLUMN-BY-COLUMN INVESTIGATION:

📊 EVIDENCE TO ANALYZE:
- Dataset: filtered_locations.gpkg
- Project Context: No project context available
- Codebase Context: No code context

🧠 SEMANTIC KNOWLEDGE FROM CODEBASE:
No semantic knowledge available.

📋 COLUMN DATA TO INVESTIGATE:
{'species_name': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': ['Haemorhous mexicanus', 'Pica pica', 'Pica pica']}, 'taxonomic_order': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 4, 'sample_values': ['Passeriformes', 'Passeriformes', 'Passeriformes']}, 'asdawas': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [8.91156072292315, 18.831008742960407, 21.63353417277567]}, 'weight_g': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [22.63545835787437, 221.9276595225436, 230.1582173720443]}, 'nocturnal_diurnal': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 2, 'sample_values': ['Diurnal', 'Diurnal', 'Diurnal']}, 'brrrrkk': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 158, 'sample_values': ['2025-07-20', '2024-12-20', '2025-01-03']}, 'diet_types': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': ["{'seeds': 1.0}", "{'insects': 0.5, 'carrion': 0.3, 'seeds': 0.2}", "{'insects': 0.5, 'carrion': 0.3, 'seeds': 0.2}"]}, 'closest_relatives': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': ["['Haemorhous purpureus', 'Haemorhous cassinii']", "['Pica hudsonia']", "['Pica hudsonia']"]}, 'latitude': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [32.346401128781864, 38.44235705433985, 43.831044771135495]}, 'longitude': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [-91.0332914317774, -123.20837942455418, -90.37110890379113]}, 'altitude_m': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [557.0704794605397, 716.3898689319495, 206.67436170278563]}, 'habitat_type': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 4, 'sample_values': ['Urban', 'Grassland', 'Grassland']}, 'weather_condition': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 5, 'sample_values': ['Cloudy', 'Windy', 'Clear']}, 'temperature_c': {'dtype': 'float64', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [19.23505378184511, 26.60126383416036, 20.507738467172867]}, 'observation_time': {'dtype': 'object', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 184, 'sample_values': ['07:55', '15:12', '11:50']}, 'geometry': {'dtype': 'geometry', 'null_count': 0, 'null_percentage': 0.0, 'unique_count': 200, 'sample_values': [<POINT (-91.033 32.346)>, <POINT (-123.208 38.442)>, <POINT (-90.371 43.831)>]}}

🔍 FORENSIC INVESTIGATION - For EACH column:

1. **NAME ANALYSIS**: Is this column name vague, cryptic, or misleading?
2. **SEMANTIC KNOWLEDGE**: Check if we have extracted definitions from Pydantic models or code comments
3. **HIDDEN MEANING**: Based on codebase context, what business logic produces these values?
4. **CROSS-REFERENCE**: Map suspicious names to potential code locations.
5. **PATTERN DETECTION**: Are there magic numbers, unusual ranges, or suspicious patterns?

🎯 EFFICIENCY REQUIREMENTS:
- Maximum response: 1200 characters total
- Use precise technical terminology
- Focus on highest-value insights

🔍 REQUIRED ANALYSIS FORMAT:
1. **DOMAIN ANALYSIS**: Overall business domain and context of this dataset
2. **DATA QUALITY**: Overall structural quality assessment
3. **COLUMN INTERPRETATIONS**: For each column, provide nested analysis
4. **BUSINESS VALUE**: How this data provides business insights

⚠️ CRITICAL Output Format Rules:
- Output must be **valid JSON only** matching DataAIEnrichment schema
- column_interpretations must contain nested ai_enrichment objects
- relationship_to_other_columns: Must be a list ["relation1", "relation2"] or null
- Never include markdown or prose
- Focus on business logic and data relationships

📋 Required Output Structure (DataAIEnrichment format):
{
  "domain_analysis": "Overall domain this dataset represents and business context",
  "data_quality_assessment": "Overall assessment of data structure and quality",
  "column_interpretations": {
    "column_name": {
      "ai_enrichment": {
        "semantic_meaning": "What this column represents",
        "domain_context": "Business domain context", 
        "usage_guidance": "How to use this column",
        "data_quality_assessment": "Quality notes",
        "relationship_to_other_columns": ["related_col1", "related_col2"]
      }
    }
  },
  "business_value_assessment": "How this data provides business value and insights"
}

📋 Return optimized JSON in DataAIEnrichment format with proper nested structure.


Context Information:
file_name: filtered_locations.gpkg
project_summary: No project context available
code_summary: No code context

columns_data:
altitude_m:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 557.0704794605397
  - 716.3898689319495
  - 206.67436170278563
  unique_count: 200
asdawas:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 8.91156072292315
  - 18.831008742960407
  - 21.63353417277567
  unique_count: 200
brrrrkk:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - '2025-07-20'
  - '2024-12-20'
  - '2025-01-03'
  unique_count: 158
closest_relatives:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - '[''Haemorhous purpureus'', ''Haemorhous cassinii'']'
  - '[''Pica hudsonia'']'
  - '[''Pica hudsonia'']'
  unique_count: 5
diet_types:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - '{''seeds'': 1.0}'
  - '{''insects'': 0.5, ''carrion'': 0.3, ''seeds'': 0.2}'
  - '{''insects'': 0.5, ''carrion'': 0.3, ''seeds'': 0.2}'
  unique_count: 5
geometry:
  dtype: geometry
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - !!python/object/apply:shapely.io.from_wkb
    - !!binary |
      AQEAAAAkrmJyIcJWwDK1R99WLEBA
  - !!python/object/apply:shapely.io.from_wkb
    - !!binary |
      AQEAAACmZ6cWVs1ewLTF7CefOENA
  - !!python/object/apply:shapely.io.from_wkb
    - !!binary |
      AQEAAABjQo8/wJdWwPbE0Kxf6kVA
  unique_count: 200
habitat_type:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Urban
  - Grassland
  - Grassland
  unique_count: 4
latitude:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 32.346401128781864
  - 38.44235705433985
  - 43.831044771135495
  unique_count: 200
longitude:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - -91.0332914317774
  - -123.20837942455418
  - -90.37110890379113
  unique_count: 200
nocturnal_diurnal:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Diurnal
  - Diurnal
  - Diurnal
  unique_count: 2
observation_time:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 07:55
  - '15:12'
  - '11:50'
  unique_count: 184
species_name:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Haemorhous mexicanus
  - Pica pica
  - Pica pica
  unique_count: 5
taxonomic_order:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Passeriformes
  - Passeriformes
  - Passeriformes
  unique_count: 4
temperature_c:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 19.23505378184511
  - 26.60126383416036
  - 20.507738467172867
  unique_count: 200
weather_condition:
  dtype: object
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - Cloudy
  - Windy
  - Clear
  unique_count: 5
weight_g:
  dtype: float64
  null_count: 0
  null_percentage: 0.0
  sample_values:
  - 22.63545835787437
  - 221.9276595225436
  - 230.1582173720443
  unique_count: 200

semantic_column_knowledge: No semantic knowledge available.

GitHub Copilot: {
  "domain_analysis":"Field ornithological point observations linking species, traits, environment and geometry",
  "data_quality_assessment":"No nulls; geometry present; nested/stringified dicts and lists; cryptic fields (asdawas,brrrrkk) need provenance",
  "column_interpretations":{
    "species_name":{"ai_enrichment":{"semantic_meaning":"Latin binomial identifier","domain_context":"taxonomy","usage_guidance":"primary grouping key","data_quality_assessment":"complete,low cardinality","relationship_to_other_columns":["taxonomic_order","closest_relatives"]}},
    "taxonomic_order":{"ai_enrichment":{"semantic_meaning":"order-level taxon","domain_context":"taxonomy","usage_guidance":"higher-level aggregation","data_quality_assessment":"complete","relationship_to_other_columns":["species_name","habitat_type"]}},
    "asdawas":{"ai_enrichment":{"semantic_meaning":"cryptic numeric index/score","domain_context":"derived morphometric/score","usage_guidance":"do not model until provenance known","data_quality_assessment":"numeric,cryptic","relationship_to_other_columns":["weight_g","altitude_m"]}},
    "weight_g":{"ai_enrichment":{"semantic_meaning":"mass (g)","domain_context":"morphology","usage_guidance":"condition/biomass metrics","data_quality_assessment":"continuous,valid","relationship_to_other_columns":["asdawas","latitude"]}},
    "nocturnal_diurnal":{"ai_enrichment":{"semantic_meaning":"activity period label","domain_context":"behavior","usage_guidance":"categorical filter","data_quality_assessment":"binary,consistent","relationship_to_other_columns":["observation_time","brrrrkk"]}},
    "brrrrkk":{"ai_enrichment":{"semantic_meaning":"observation date (string)","domain_context":"temporal","usage_guidance":"use as observation_date after parsing","data_quality_assessment":"dates parsed as strings","relationship_to_other_columns":["observation_time","species_name"]}},
    "diet_types":{"ai_enrichment":{"semantic_meaning":"diet composition (stringified dict)","domain_context":"ecology","usage_guidance":"parse and normalize proportions","data_quality_assessment":"nested string,redundant forms possible","relationship_to_other_columns":["closest_relatives","species_name"]}},
    "closest_relatives":{"ai_enrichment":{"semantic_meaning":"stringified list of taxa","domain_context":"phylogeny","usage_guidance":"use for similarity/validation","data_quality_assessment":"stringified list,low cardinality","relationship_to_other_columns":["species_name","diet_types"]}},
    "latitude":{"ai_enrichment":{"semantic_meaning":"decimal latitude","domain_context":"geography","usage_guidance":"map and spatial joins","data_quality_assessment":"valid floats","relationship_to_other_columns":["longitude","geometry"]}},
    "longitude":{"ai_enrichment":{"semantic_meaning":"decimal longitude","domain_context":"geography","usage_guidance":"map and spatial joins","data_quality_assessment":"valid floats","relationship_to_other_columns":["latitude","geometry"]}},
    "altitude_m":{"ai_enrichment":{"semantic_meaning":"elevation (m)","domain_context":"environment","usage_guidance":"environmental covariate","data_quality_assessment":"continuous,valid","relationship_to_other_columns":["temperature_c","asdawas"]}},
    "habitat_type":{"ai_enrichment":{"semantic_meaning":"habitat category","domain_context":"ecology","usage_guidance":"stratify analyses","data_quality_assessment":"categorical,4 values","relationship_to_other_columns":["species_name","weather_condition"]}},
    "weather_condition":{"ai_enrichment":{"semantic_meaning":"short weather label","domain_context":"environment","usage_guidance":"covariate for behavior","data_quality_assessment":"categorical","relationship_to_other_columns":["temperature_c","observation_time"]}},
    "temperature_c":{"ai_enrichment":{"semantic_meaning":"air temperature °C","domain_context":"environment","usage_guidance":"covariate","data_quality_assessment":"continuous,valid","relationship_to_other_columns":["weather_condition","altitude_m"]}},
    "observation_time":{"ai_enrichment":{"semantic_meaning":"time HH:MM","domain_context":"temporal","usage_guidance":"combine with date for timestamp","data_quality_assessment":"string,high cardinality","relationship_to_other_columns":["brrrrkk","nocturnal_diurnal"]}},
    "geometry":{"ai_enrichment":{"semantic_meaning":"point geometry (WKB)","domain_context":"spatial","usage_guidance":"primary spatial key for mapping","data_quality_assessment":"valid geometries","relationship_to_other_columns":["latitude","longitude"]}}
  },
  "business_value_assessment":"Enables spatial ecology, species distribution and trait-based analyses; clarify cryptic/derived fields and parse nested diet/relative fields for reliable models"
}

raymondpeterdorn: 🧠 ROLE: Data Forensics Analyst

You are an expert data analyst investigating tabular datasets to uncover data meanings,
quality patterns, and business logic from structural evidence.
Think like a **data archaeologist**:
- Infer dataset purpose from structure and column patterns
- Detect business rules and data validation logic from constraints
- Identify data quality issues and transformation requirements  
- Prioritize precision, brevity, and factual reasoning

Your output will populate a predefined YAML schema describing the dataset analysis.


🕵️ TASK: Conduct targeted dataset analysis based on provided evidence.

📊 Dataset Context:
- Dataset: filtered_locations.gpkg
- Structure: 200 rows x 16 columns
- Project: No project context available

⚙️ Schema Reference (summary only):
- ai_interpretation: any
- ai_confidence: any
- ai_domain_context: any
- usage_guidance: any
- hidden_meaning: any  # Explanations, business logic, or context buried in code comments, variable names...
- suspicious_patterns: any  # List of oddities found: poorly named variables/columns, magic numbers, confusing...
- cross_references: any  # Mapping of confusing field/variable names to where in the codebase they are crea...
- detective_insights: any  # Reverse-engineered understanding of developer intent, business requirements, or ...
- domain_analysis: any  # The domain this data represents and its key characteristics in a business contex...
- data_quality_assessment: any  # Assessment of how well-structured and clear the overall schema is.
- column_interpretations: any  # Detailed interpretations of each column in the dataset.
- business_value_assessment: any  # How this data might provide business value and insights.

🎯 Analysis Guidelines:
1. Use only verifiable evidence from dataset structure and column information.
2. Infer business meaning and data quality — do not restate technical details verbatim.
3. When information is missing, leave the corresponding field as an empty string.
4. Use concise, factual descriptions suitable for data governance metadata.
5. Avoid verbose speculation, repetition, or narrative commentary.

📏 Efficiency Rules:
- Total output ≤ 1200 characters
- Each field ≤ 200 characters
- Avoid redundant phrasing or filler words

⚠️ CRITICAL Output Format Rules:
- ai_confidence: Must be exactly "LOW", "MEDIUM", or "HIGH" (not descriptive text)
- suspicious_patterns: Must be a list ["pattern1", "pattern2"] or null
- cross_references: Must be a dict {"field": "explanation"} or null  
- column_interpretations: Must be a dict with column names as keys, each containing:
  {"semantic_meaning": "...", "domain_context": "...", "usage_guidance": "...", "data_quality_assessment": "..."}
- Never include markdown or prose — output must be **valid JSON only**
- Use technical language to explain *why* columns exist, not *what* they contain
- Focus on business logic and data relationships
- Skip commentary about the task or user

� Example Output Structure:
{
  "ai_interpretation": "Brief dataset purpose",
  "ai_confidence": "HIGH",
  "ai_domain_context": "Domain area",
  "usage_guidance": "How to use this data", 
  "hidden_meaning": "Hidden business logic",
  "suspicious_patterns": ["pattern1", "pattern2"],
  "cross_references": {"field": "explanation"},
  "detective_insights": "Key insights",
  "domain_analysis": "Domain analysis", 
  "data_quality_assessment": "Quality assessment",
  "column_interpretations": {
    "column_name": {
      "semantic_meaning": "What this column represents",
      "domain_context": "Business domain context",
      "usage_guidance": "How to use this column",
      "data_quality_assessment": "Quality notes"
    }
  },
  "business_value_assessment": "Business value"
}

�💡 Output Format:
Return structured JSON matching the injected schema class.

Return valid JSON only.


Context Information:
file_name: filtered_locations.gpkg
rows: 200
num_columns: 16
project_summary: No project context available

columns:
- species_name
- taxonomic_order
- asdawas
- weight_g
- nocturnal_diurnal
- brrrrkk
- diet_types
- closest_relatives
- latitude
- longitude
- altitude_m
- habitat_type
- weather_condition
- temperature_c
- observation_time
- geometry


GitHub Copilot: {
  "ai_interpretation":"Point-level ornithological observations for trait, temporal and spatial analyses",
  "ai_confidence":"MEDIUM",
  "ai_domain_context":"Ornithology / spatial ecology",
  "usage_guidance":"Use for distribution, trait and behavior models; parse stringified nested fields and resolve cryptic indices before modeling",
  "hidden_meaning":"asdawas likely derived index; brrrrkk stores observation date; diet_types and closest_relatives are serialized complex objects",
  "suspicious_patterns":["cryptic names (asdawas,brrrrkk)","stringified nested structures (diet_types,closest_relatives)","redundant temporal representations (brrrrkk,observation_time)"],
  "cross_references":null,
  "detective_insights":"Dataset merges raw measures (weight) with pipeline-derived indices and serialized complex types; intended for ecological modeling but needs normalization",
  "domain_analysis":"Species-level field observations with taxonomy, morphometrics, environment and precise geometry",
  "data_quality_assessment":"No nulls; geometries present; several fields require parsing/ provenance to avoid misuse",
  "column_interpretations":{
    "species_name":{"semantic_meaning":"Latin binomial identifier","domain_context":"taxonomy","usage_guidance":"primary grouping key for joins/aggregates","data_quality_assessment":"complete,low cardinality"},
    "taxonomic_order":{"semantic_meaning":"order-level taxon","domain_context":"taxonomy","usage_guidance":"higher-level aggregation and filtering","data_quality_assessment":"complete"},
    "asdawas":{"semantic_meaning":"cryptic numeric index/score","domain_context":"derived morphometric/score","usage_guidance":"investigate provenance before use","data_quality_assessment":"numeric but undocumented"},
    "weight_g":{"semantic_meaning":"mass in grams","domain_context":"morphology","usage_guidance":"condition and biomass metrics","data_quality_assessment":"continuous,valid"},
    "nocturnal_diurnal":{"semantic_meaning":"activity period label","domain_context":"behavior","usage_guidance":"categorical filter; derive boolean if needed","data_quality_assessment":"binary,consistent"},
    "brrrrkk":{"semantic_meaning":"observation date (string)","domain_context":"temporal","usage_guidance":"parse to date and combine with observation_time","data_quality_assessment":"string dates,158 uniques"},
    "diet_types":{"semantic_meaning":"diet composition (serialized dict)","domain_context":"ecology","usage_guidance":"parse to normalized proportions","data_quality_assessment":"nested,stringified"},
    "closest_relatives":{"semantic_meaning":"list of similar taxa (serialized)","domain_context":"phylogeny","usage_guidance":"parse for validation or similarity joins","data_quality_assessment":"stringified lists"},
    "latitude":{"semantic_meaning":"decimal latitude","domain_context":"geography","usage_guidance":"spatial joins and mapping","data_quality_assessment":"valid floats"},
    "longitude":{"semantic_meaning":"decimal longitude","domain_context":"geography","usage_guidance":"spatial joins and mapping","data_quality_assessment":"valid floats"},
    "altitude_m":{"semantic_meaning":"elevation in meters","domain_context":"environment","usage_guidance":"environmental covariate","data_quality_assessment":"continuous,valid"},
    "habitat_type":{"semantic_meaning":"habitat category","domain_context":"ecology","usage_guidance":"stratify analyses","data_quality_assessment":"categorical,low cardinality"},
    "weather_condition":{"semantic_meaning":"short weather label","domain_context":"environment","usage_guidance":"use as covariate with temperature","data_quality_assessment":"categorical"},
    "temperature_c":{"semantic_meaning":"air temperature °C","domain_context":"environment","usage_guidance":"environmental covariate","data_quality_assessment":"continuous,valid"},
    "observation_time":{"semantic_meaning":"time of day HH:MM","domain_context":"temporal","usage_guidance":"combine with brrrrkk for timestamp","data_quality_assessment":"string,high cardinality"},
    "geometry":{"semantic_meaning":"point geometry (WKB)","domain_context":"spatial","usage_guidance":"primary spatial key for mapping and spatial joins","data_quality_assessment":"valid geometries"}
  },
  "business_value_assessment":"Supports spatial ecology, species distribution and trait modeling; cleanup (parse, normalize, document derived indices) needed for reproducible insights"
}

raymondpeterdorn: Analyze this image file and provide insights that fit within these STRICT LIMITS:\n\nProvide insights that fit within these STRICT LIMITS:

RESPONSE SIZE CONSTRAINTS:
- Total response must be under 2400 characters
- Each field value: maximum 200 characters
- Use concise, technical language
- Focus on accuracy over completeness
- image: 146 bytes

FIELD-SPECIFIC CONSTRAINTS:
- ai_interpretation: One sentence summary only
- ai_confidence: HIGH/MEDIUM/LOW only
- ai_domain_context: Domain + key characteristics
- usage_guidance: 1-2 practical applications
- hidden_meaning: Key insights only
- suspicious_patterns: Main issues only
- cross_references: Essential connections only
- content_description: Visual/audio content (100 chars max)
- quality_assessment: Technical quality + defects
- technical_analysis: Format details + compression
- use_case_recommendations: Likely purpose + applications
- applications: Potential domains + usage

EXPECTED SIZE: Target ~1200 characters total.

Omit optional fields if approaching size limits.

CONTEXT DATA:
{
  "file_name": "pixel_bird.png",
  "file_path": "bird_demo/output/companion/pixel_bird.png",
  "media_metadata": {
    "media_type": "image",
    "file_size_bytes": 146,
    "media_info": {
      "duration": null,
      "dimensions": [
        20,
        20
      ],
      "sample_rate": null,
      "bit_rate": null,
      "channels": null,
      "color_space": "RGB"
    }
  },
  "file_size_mb": 0.0001392364501953125,
  "semantic_knowledge": "No semantic knowledge extracted from codebase."
}

REQUIRED JSON SCHEMA:
{
  "$defs": {
    "ConfidenceLevel": {
      "description": "Confidence levels for AI-generated content.",
      "enum": [
        "LOW",
        "MEDIUM",
        "HIGH"
      ],
      "title": "ConfidenceLevel",
      "type": "string"
    }
  },
  "description": "AI-generated insights about media files.",
  "properties": {
    "ai_interpretation": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Ai Interpretation"
    },
    "ai_confidence": {
      "anyOf": [
        {
          "$ref": "#/$defs/ConfidenceLevel"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
    "ai_domain_context": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Ai Domain Context"
    },
    "usage_guidance": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Usage Guidance"
    },
    "hidden_meaning": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Explanations, business logic, or context buried in code comments, variable names, or logic that clarify what this really represents beyond surface appearance.",
      "title": "Hidden Meaning"
    },
    "suspicious_patterns": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "List of oddities found: poorly named variables/columns, magic numbers, confusing transformations, misleading names, or hacky implementations.",
      "title": "Suspicious Patterns"
    },
    "cross_references": {
      "anyOf": [
        {
          "additionalProperties": {
            "type": "string"
          },
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Mapping of confusing field/variable names to where in the codebase they are created, defined, calculated, or explained.",
      "title": "Cross References"
    },
    "detective_insights": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Reverse-engineered understanding of developer intent, business requirements, or domain knowledge that explains why this exists and how it connects to the larger system.",
      "title": "Detective Insights"
    },
    "content_description": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Detailed description of the media content, including subjects, themes, or audio content.",
      "title": "Content Description"
    },
    "quality_assessment": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Assessment of the media quality, including resolution, clarity, noise levels, or compression artifacts.",
      "title": "Quality Assessment"
    },
    "technical_analysis": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Technical analysis of the media file, including encoding, format characteristics, or technical issues.",
      "title": "Technical Analysis"
    },
    "use_case_recommendations": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Recommended use cases or applications for the media file based on its content and quality.",
      "title": "Use Case Recommendations"
    },
    "applications": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Potential applications or domains where this media file would be useful or relevant.",
      "title": "Applications"
    },
    "processing_recommendations": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Recommendations for processing or enhancing the media file to improve its quality or usability.",
      "title": "Processing Recommendations"
    }
  },
  "title": "MediaAIEnrichment",
  "type": "object"
}

INSTRUCTIONS:
1. Analyze the provided context data thoroughly and extract all relevant information
2. Generate a detailed JSON response that exactly matches the schema above
3. Fill all fields with comprehensive, detailed information based on the context
4. For missing information:
   - Use null for optional object/array fields
   - Use empty string "" for string fields where information is not available
   - Never use null for string values in dictionaries
   - For dict fields, provide detailed key-value pairs even when specific data is limited
   - For list fields, provide comprehensive lists with detailed entries
5. Ensure all enum values match exactly as specified
6. When describing data columns, provide thorough explanations of each column's meaning, purpose, and content
7. For model analysis, include detailed assessments of model capabilities, limitations, and suggested uses
8. Return ONLY valid JSON - no markdown formatting or additional text


GitHub Copilot: {
  "ai_interpretation": "20×20 RGB pixel-art bird icon, symbolic/illustrative rather than photographic.",
  "ai_confidence": "HIGH",
  "ai_domain_context": "Iconography / pixel-art sprites; small raster thumbnails for UI or demo datasets.",
  "usage_guidance": "Use as UI thumbnail, sprite or dataset exemplar; preserve nearest-neighbor scaling for clarity.",
  "hidden_meaning": "Likely programmatically generated/exported as a minimal sprite for companion output; optimized for bytes and legibility.",
  "suspicious_patterns": [
    "very low resolution (20×20)",
    "tiny file size (146 bytes) may indicate palette optimization or stripped metadata",
    "limited color palette"
  ],
  "cross_references": {
    "pixel_bird.png": "bird_demo/output/companion/pixel_bird.png (companion/demo output asset)"
  },
  "detective_insights": "Asset appears produced by demo pipeline for compact delivery; not suitable for analytic image tasks without upscaling.",
  "content_description": "Stylized single bird sprite, flat colors, no photographic detail.",
  "quality_assessment": "Technically intact but extremely low resolution; visible blockiness when scaled; minimal metadata.",
  "technical_analysis": "PNG raster, 20×20 px, RGB, 146 B; likely uses small palette and compression; no audio stream.",
  "use_case_recommendations": [
    "UI/iconography",
    "low-bandwidth demos",
    "game sprite or avatar"
  ],
  "applications": [
    "mobile/web UI",
    "tutorials/demos",
    "retro-style games"
  ],
  "processing_recommendations": "For larger uses: nearest-neighbor upscaling, optional palette conversion to PNG8, embed provenance metadata, keep original for pixel-art fidelity."
}