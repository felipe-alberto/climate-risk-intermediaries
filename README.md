# Climate Risk Intermediaries Dataset

This repository contains a hand collected dataset on **sovereign climate risk insurance facilities**.
The data set focuses on regional risk pools and insurance facilities involved in climate-risk sharing,
insurance, and disaster financing.

The dataset is intended for use in **academic research**.

---

## Contents

- `data/intermediaries.csv`  
    List of climate-risk intermediaries in the dataset.

- `data/sovereigns.csv`  
    List of sovereigns participating in climate-risk intermediaries
    in the dataset.

- `data/intermediary_panel_data.csv`  
    Yearly observations on several parameters related to 
    the financial and operational performance of climate-risk 
    intermediaries.

- `README.md`  
    Documentation and usage notes.

- `LICENSE`  
    Licensing information for the repository.

---

## Data Description

- `data/intermediary_panel_data.csv`  
Each row corresponds to a time-indexed observation of a single variable related to a climate-risk intermediary.
Each column specifies a property of the time-indexed observation.

Current variables include (non-exhaustive):
- reported_events
- parametric_insurance_contracts_income
- parametric_insurance_claims_paid
- parametric_insurance_claims_recovered
- parametric_insurance_contracts_aggregate_coverage
- parametric_insurance_first_loss_retention
- parametric_insurance_reinsurance_tranche
- parametric_insurance_excess_loss_cutoff

Current properties include:
- intermediary_id
- intermediary_name
- program_id
- program_name
- variable_code
- value
- unit
- measure_type
- policy_year_start
- policy_year_end

A formal data dictionary may be added in future versions.

---

## Usage

The dataset is provided in CSV format and can be loaded in most statistical and
data-analysis environments.
