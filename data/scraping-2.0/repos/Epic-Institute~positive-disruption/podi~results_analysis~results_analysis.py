# region
import os

import numpy as np
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# endregion


def results_analysis(
    scenario,
    energy_output,
    afolu_output,
    emissions_output,
    emissions_output_co2e,
    climate_output_concentration,
    climate_output_temperature,
    climate_output_forcing,
    data_start_year,
    data_end_year,
    proj_end_year,
):
    ###############################
    #  TECHNOLOGY ADOPTION RATES  #
    ###############################

    # region

    # Load external dataset of technology adoption rates

    analog = pd.read_csv(
        "podi/data/external/CHATTING_SPLICED.csv",
        # usecols=["variable", "label", "iso3c", "year", "category", "value"],
    )

    labels = [
        "Combine harvesters - threshers in use",
        "Land agricultural land area 1000 ha",
        "Agricultural tractors in use",
        "Total vehicles (OICA)",
        "Aluminum primary production, in metric tons",
        "Land arable land area 1000 ha",
        "ATMs",
        "Air transport, passengers carried",
        "Civil aviation passenger-KM traveled",
        "Civil aviation ton-KM of cargo carried",
        "Households that subscribe to cable",
        "Cellular subscriptions",
        "Personal computers",
        # "Electricity from coal (TWH)",
        # "Electric power consumption (KWH)",
        # "Electricity from gas (TWH)",
        # "Electricity from hydro (TWH)",
        # "Electricity from nuclear (TWH)",
        # "Electricity from oil (TWH)",
        # "Electricity from other renewables (TWH)",
        # "Electricity from solar (TWH)",
        # "Electricity from wind (TWH)",
        # "Gross output of electric energy (TWH)",
        # "Electricity Generating Capacity, 1000 kilowatts",
        "Fertilizer ammonium nitrate (AN) agricultural use tonnes",
        "Fertilizer ammonium sulphate agricultural use tonnes",
        "Fertilizer diammonium phosphate (DAP) agricultural use tonnes",
        "Fertilizer potassium chloride (muriate of potash) (MOP) agricultural use tonnes",
        "Fertilizer NPK fertilizers agricultural use tonnes",
        "Fertilizer other NP compounds agricultural use tonnes",
        "Fertilizer superphosphates above 35 percent agricultural use tonnes",
        "Fertilizer potassium sulphate (sulphate of potash) (SOP) agricultural use tonnes",
        "Aggregate kg of fertilizer consumed",
        "Fertilizer urea agricultural use tonnes",
        "Land naturally regenerating forest area 1000 ha",
        "Land planted forest area 1000 ha",
        "People with internet access",
        "Area equipped to provide water to crops",
        "Automatic looms",
        "Ordinary and automatic looms",
        "Items mailed or received",
        # "% Arable land share in agricultural land",
        # "% Irrigated area as a share of cultivated land",
        "Pesticide fungicides and bactericides agricultural use tonnes",
        "Pesticide herbicides agricultural use tonnes",
        "Pesticide insecticides agricultural use tonnes",
        "Pesticide mineral oils agricultural use tonnes",
        "Pesticide other pesticides nes agricultural use tonnes",
        "Pesticide rodenticides agricultural use tonnes",
        "Total metric tons of pesticides in agricultural use",
        "Radios",
        "Geographical/route lengths of line open at the end of the year",
        "Rail lines (total route-km)",
        "Thousands of passenger journeys by railway",
        "Passenger journeys by railway (passenger-km)",
        "Metric tons of freight carried on railways (excluding livestock and passenger baggage)",
        "Freight carried on railways (excluding livestock and passenger baggage) (ton-km)",
        "Length of Paved Road (km)",
        "Secure internet servers",
        "Ships of all kinds",
        "Tonnage of ships of all kinds",
        "Tonnage of sail ships",
        "Tonnage of steam ships",
        "Tonnage of steam and motor ships",
        "Mule spindles",
        "Ring spindles",
        "Steel demand in thousand metric tons",
        "Steel production in thousand metric tons",
        "Stainless steel production",
        "Telegrams",
        "Fixed telephone subscriptions",
        "Television sets",
        "Weight of artificial fibers in spindles",
        "Weight of other fibers in spindles",
        "Weight of synthetic fibers in spindles",
        "Weight of all fibers in spindles",
        "Passenger car vehicles",
        "Passenger cars (BTS)",
        "Commercial vehicles (bus, taxi)",
        "Commercial vehicles (BTS)",
        "Total vehicles (BTS)",
    ]

    # create dataframe of analog where label is in labels
    analog = analog.loc[analog["label"].isin(labels)]

    # reformat to match energy_output
    analog = analog.rename(
        columns={
            "variable": "flow_short",
            "label": "flow_long",
            "iso3c": "region",
            "year": "year",
            "category": "flow_category",
            "value": "value",
        }
    )
    # make year column rows
    analog = analog.pivot_table(
        index=[
            "region",
            "flow_category",
            "flow_long",
            "flow_short",
        ],
        columns="year",
        values="value",
    )
    # add index for model and scenario
    analog = analog.assign(
        model="PD22",
        scenario="baseline",
        sector="Energy",
        product_category="Technology",
        product_long="Technology",
        product_short="Technology",
        unit="multiple (hover over a time series to see units)",
    )
    # replace region ISO codes with region names from the 'WEB Region Lower' column of region_categories.csv
    region_categories = pd.read_csv(
        "podi/data/region_categories.csv",
        usecols=["WEB Region Lower", "ISO"],
    )
    analog = analog.reset_index().merge(
        region_categories[["WEB Region Lower", "ISO"]],
        left_on="region",
        right_on="ISO",
    )

    # drop region column and rename WEB Region Lower to region
    analog = analog.drop(columns=["region", "ISO"])
    analog = analog.rename(columns={"WEB Region Lower": "region"})
    # drop rows where region is NaN
    analog = analog.dropna(subset=["region"])
    analog = analog.set_index(
        [
            "model",
            "scenario",
            "region",
            "sector",
            "product_category",
            "product_long",
            "product_short",
            "flow_category",
            "flow_long",
            "flow_short",
            "unit",
        ]
    )

    # concat a copy of analog to analog, except 'baseline' is replaced with 'pathway'
    analog_pathway = analog.copy()
    analog_pathway = analog_pathway.reset_index()
    analog_pathway["scenario"] = "pathway"
    analog_pathway = analog_pathway.set_index(
        [
            "model",
            "scenario",
            "region",
            "sector",
            "product_category",
            "product_long",
            "product_short",
            "flow_category",
            "flow_long",
            "flow_short",
            "unit",
        ]
    )
    analog = pd.concat([analog, analog_pathway])

    analog.columns = analog.columns.astype(int)

    # list duplicate rows in pd.concat([analog, afolu_output, energy_output], axis=1, join="outer")
    # Concatenate the DataFrames

    afolu_output[
        (afolu_output.index.duplicated())
    ].reset_index().flow_short.unique()

    # Combine analog, afolu_output and energy_output, using the same columns when available
    technology_adoption_output = pd.concat(
        [analog, afolu_output, energy_output]
    )

    # reduce memory usage
    technology_adoption_output = technology_adoption_output.astype("float32")

    # save to parquet, update columns to be strings
    technology_adoption_output.columns = (
        technology_adoption_output.columns.astype(str)
    )

    # list duplicate values in technology_adoption_output.columns
    duplicate_columns = technology_adoption_output.columns[
        technology_adoption_output.columns.duplicated()
    ]

    technology_adoption_output.to_parquet(
        "podi/data/technology_adoption_output.parquet", compression="brotli"
    )
    """
    # Percent of electric power that is renewables
    electricity = (
        energy_output.loc[
            slice(None),
            slice(None),
            slice(None),
            ["Electric Power"],
            slice(None),
            slice(None),
            [
                "GEOTHERM",
                "HYDRO",
                "SOLARPV",
                "ROOFTOP",
                "SOLARTH",
                "OFFSHORE",
                "ONSHORE",
                "TIDE",
                "NUCLEAR",
            ],
            ["Electricity output"],
        ]
        .groupby(
            [
                "model",
                "scenario",
                "region",
                "sector",
                "product_long",
            ],
            observed=True,
        )
        .sum(numeric_only=True)
        .parallel_apply(
            lambda x: x.divide(
                energy_output.loc[
                    slice(None),
                    slice(None),
                    slice(None),
                    ["Electric Power"],
                    slice(None),
                    slice(None),
                    slice(None),
                    ["Electricity output"],
                ]
                .groupby(["model", "scenario", "region"], observed=True)
                .sum(numeric_only=True)
                .loc[x.name[0], x.name[1], x.name[2]]
            ),
            axis=1,
        )
    )

    # Percent of transport energy that is electric or nonelectric renewables
    transport = (
        energy_output.loc[
            slice(None),
            slice(None),
            slice(None),
            ["Transportation"],
            slice(None),
            slice(None),
            ["ELECTR", "HYDROGEN"],
            slice(None),
            [
                "Road – 2&3-wheel",
                "Road – Buses&Vans",
                "Road – Light-duty vehicles",
                "Road – Trucks",
                "Rail – Heavy-duty",
                "Rail – Light-duty",
                "Transport not elsewhere specified",
                "Domestic navigation",
                "International marine bunkers",
                "Domestic aviation – Long-range",
                "Domestic aviation – Short-range",
                "International aviation bunkers",
                "Non-energy use in transport",
                "Pipeline transport",
                "Losses",
                "Memo: Non-energy use in transport equipment",
            ],
        ]
        .groupby(
            ["model", "scenario", "region", "sector", "flow_long"],
            observed=True,
        )
        .sum(numeric_only=True)
        .parallel_apply(
            lambda x: x.divide(
                energy_output.loc[
                    slice(None),
                    slice(None),
                    slice(None),
                    ["Transportation"],
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    [
                        "Road – 2&3-wheel",
                        "Road – Buses&Vans",
                        "Road – Light-duty vehicles",
                        "Road – Trucks",
                        "Rail – Heavy-duty",
                        "Rail – Light-duty",
                        "Transport not elsewhere specified",
                        "Domestic navigation",
                        "International marine bunkers",
                        "Domestic aviation – Long-range",
                        "Domestic aviation – Short-range",
                        "International aviation bunkers",
                        "Non-energy use in transport",
                        "Pipeline transport",
                        "Losses",
                        "Memo: Non-energy use in transport equipment",
                    ],
                ]
                .groupby(["model", "scenario", "region"], observed=True)
                .sum(numeric_only=True)
                .loc[x.name[0], x.name[1], x.name[2]]
            ),
            axis=1,
        )
    )

    # Percent of buildings energy that is electric or nonelectric renewables
    buildings = (
        energy_output.loc[
            slice(None),
            slice(None),
            slice(None),
            ["Commercial", "Residential"],
            slice(None),
            slice(None),
            ["ELECTR", "SOLARTH", "MUNWASTER", "GEOTHERM"],
            slice(None),
            slice(None),
            ["RESIDENT", "COMMPUB"],
        ]
        .groupby(
            ["model", "scenario", "region", "sector", "flow_long"],
            observed=True,
        )
        .sum(numeric_only=True)
        .parallel_apply(
            lambda x: x.divide(
                energy_output.loc[
                    slice(None),
                    slice(None),
                    slice(None),
                    ["Commercial", "Residential"],
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                ]
                .groupby(
                    ["model", "scenario", "region", "sector"], observed=True
                )
                .sum(numeric_only=True)
                .loc[x.name[0], x.name[1], x.name[2], x.name[3]]
            ),
            axis=1,
        )
    )
    buildings = (
        buildings.rename(
            index={"Commercial": "Buildings", "Residential": "Buildings"}
        )
        .groupby(buildings.index.names, observed=True)
        .sum(numeric_only=True)
    )

    # Percent of industry energy that is electric or nonelectric renewables
    industry = (
        energy_output.loc[
            slice(None),
            slice(None),
            slice(None),
            ["Industrial"],
            slice(None),
            slice(None),
            ["ELECTR", "SOLARTH", "HYDROGEN", "MUNWASTER", "GEOTHERM"],
            "Final consumption",
            [
                "Machinery",
                "Non-ferrous metals",
                "Final consumption not elsewhere specified",
                "Food and tobacco",
                "Agriculture/forestry",
                "Non-metallic minerals",
                "Chemical and petrochemical",
                "Iron and steel",
                "Industry not elsewhere specified",
            ],
        ]
        .groupby(
            ["model", "scenario", "region", "sector", "flow_long"],
            observed=True,
        )
        .sum(numeric_only=True)
        .parallel_apply(
            lambda x: x.divide(
                energy_output.loc[
                    slice(None),
                    slice(None),
                    slice(None),
                    ["Industrial"],
                    slice(None),
                    slice(None),
                    slice(None),
                    "Final consumption",
                ]
                .groupby(
                    ["model", "scenario", "region", "sector"], observed=True
                )
                .sum(numeric_only=True)
                .loc[x.name[0], x.name[1], x.name[2], x.name[3]]
            ),
            axis=1,
        )
        .sort_values(by=[2050], axis=0)
    )

    industry_other = (
        energy_output.loc[
            slice(None),
            slice(None),
            slice(None),
            ["Industrial"],
            slice(None),
            slice(None),
            ["ELECTR", "SOLARTH", "HYDROGEN", "MUNWASTER", "GEOTHERM"],
            "Final consumption",
            [
                "Paper, pulp, and print",
                "Fishing",
                "Wood and wood products",
                "Transport equipment",
                "Textile and leather",
                "Construction",
                "Mining and quarrying",
            ],
        ]
        .groupby(["model", "scenario", "region", "sector"], observed=True)
        .sum(numeric_only=True)
        .parallel_apply(
            lambda x: x.divide(
                energy_output.loc[
                    slice(None),
                    slice(None),
                    slice(None),
                    ["Industrial"],
                    slice(None),
                    slice(None),
                    slice(None),
                    "Final consumption",
                    slice(None),
                ]
                .groupby(
                    ["model", "scenario", "region", "sector"], observed=True
                )
                .sum(numeric_only=True)
                .loc[x.name[0], x.name[1], x.name[2], x.name[3]]
            ),
            axis=1,
        )
        .sort_values(by=[2050], axis=0)
    )

    industry_other = pd.concat(
        [industry_other], keys=["Other"], names=["flow_long"]
    ).reorder_levels(["model", "scenario", "region", "sector", "flow_long"])

    industry = pd.concat([industry, industry_other])

    # Percent of agriculture mitigation compared to max extent
    agriculture = (
        afolu_output.loc[
            slice(None), slice(None), slice(None), ["Agriculture"]
        ]
        .groupby(
            ["model", "scenario", "region", "sector", "product_long"],
            observed=True,
        )
        .sum(numeric_only=True)
        .parallel_apply(
            lambda x: x.divide(
                afolu_output.loc[
                    slice(None), slice(None), slice(None), ["Agriculture"]
                ]
                .groupby(
                    ["model", "scenario", "region", "sector", "product_long"],
                    observed=True,
                )
                .sum(numeric_only=True)
                .loc[x.name]
                .max()
            ).fillna(0),
            axis=1,
        )
    )

    # Percent of forests & wetlands mitigation compared to max extent
    forestswetlands = (
        afolu_output.loc[
            slice(None), slice(None), slice(None), ["Forests & Wetlands"]
        ]
        .groupby(
            ["model", "scenario", "region", "sector", "product_long"],
            observed=True,
        )
        .sum(numeric_only=True)
        .parallel_apply(
            lambda x: x.divide(
                afolu_output.loc[
                    slice(None),
                    slice(None),
                    slice(None),
                    ["Forests & Wetlands"],
                ]
                .groupby(
                    ["model", "scenario", "region", "sector", "product_long"],
                    observed=True,
                )
                .sum(numeric_only=True)
                .loc[x.name]
                .max()
            ).fillna(0),
            axis=1,
        )
    )

    # Combine all verticals
    adoption_output_projections = pd.concat(
        [
            electricity,
            transport,
            buildings,
            industry,
            agriculture,
            forestswetlands,
        ]
    ).multiply(100)

    adoption_output_projections["unit"] = "% Adoption"
    adoption_output_projections.set_index("unit", append=True, inplace=True)

    adoption_output_projections.to_csv(
        "podi/data/adoption_output_projections.csv"
    )
    """
    # endregion

    ###################
    #  BIOCHAR STUDY  #
    ###################

    # region

    # load biochar study data
    # biochar_study = pd.read_csv("podi/data/APL/apl_biochar_projections.csv")
    # biochar_study.set_index(
    #     [
    #         "model",
    #         "scenario",
    #         "region",
    #         "sector",
    #         "product_category",
    #         "product_long",
    #         "product_short",
    #         "flow_category",
    #         "flow_long",
    #         "flow_short",
    #         "unit",
    #     ],
    #     inplace=True,
    # )

    # biochar_study.columns = biochar_study.columns.astype(int)

    # emissions_output_co2e.update(biochar_study.astype(emissions_output_co2e.dtypes))

    # To estimate the number of CharPallets, first filter emissions_output_co2e for all biochar flows
    emissions_output_co2e_biochar = emissions_output_co2e.loc[
        (
            emissions_output_co2e.reset_index().flow_long.isin(
                [
                    "Biochar as Ag Soil Amendment",
                    "Biochar for Carbon Removal & Sequestration",
                    "Biochar for Water Treatment",
                    "Biochar as Activated Carbon",
                    "Biochar for Construction Materials",
                ]
            )
        ).values
    ]

    # change flow_long index values from 'Biochar' to "CharPallets"
    emissions_output_co2e_biochar.reset_index(inplace=True)
    emissions_output_co2e_biochar.loc[:, "flow_long"] = "CharPallets"
    emissions_output_co2e_biochar.set_index(
        [
            "model",
            "scenario",
            "region",
            "sector",
            "product_category",
            "product_long",
            "product_short",
            "flow_category",
            "flow_long",
            "flow_short",
            "unit",
        ],
        inplace=True,
    )

    # change units from Mt CO2e to CharPallets
    mtco2e_per_charpallet = 22e-6
    charpallet_adoption = (
        emissions_output_co2e_biochar / -mtco2e_per_charpallet
    )

    # add charpallet_adoption to adoption_output_projections
    technology_adoption_output.columns = (
        technology_adoption_output.columns.astype(int)
    )
    technology_adoption_output = pd.concat(
        [technology_adoption_output, charpallet_adoption]
    )

    # reduce memory usage
    technology_adoption_output = technology_adoption_output.astype("float32")

    # save to parquet, update columns to be strings
    technology_adoption_output.columns = (
        technology_adoption_output.columns.astype(str)
    )
    technology_adoption_output.to_parquet(
        "podi/data/technology_adoption_output.parquet", compression="brotli"
    )

    # endregion

    ##################
    #  ESTIMATE CDR  #
    ##################

    # region

    # load historical CDR data from IEA https://www.iea.org/data-and-statistics/data-product/ccus-projects-database
    emissions_cdr = pd.read_excel(
        "podi/data/IEA/Other/IEA CCUS Projects Database 2023.xlsx",
        sheet_name="CCUS Projects Database",
        usecols="B,G,I,K,L,M",
        header=0,
        names=[
            "region",
            "start year",
            "status",
            "Mt low",
            "Mt high",
            "technology",
        ],
        engine="openpyxl",
    ).dropna(how="all")

    # add model, scenario, sector, product_category, product_long, product_short, flow_category, flow_long, flow_short, unit
    emissions_cdr["model"] = "PD22"
    emissions_cdr["scenario"] = "baseline"
    emissions_cdr["sector"] = "CDR"
    emissions_cdr["product_category"] = "CDR"
    emissions_cdr["product_long"] = "CO2"
    emissions_cdr["product_short"] = "CO2"
    emissions_cdr["flow_category"] = "Carbon Dioxide Removal"
    emissions_cdr["flow_long"] = "Carbon Dioxide Removal"
    emissions_cdr["flow_short"] = "CDR"
    emissions_cdr["unit"] = "Mt"
    for year in range(data_start_year, proj_end_year + 1):
        emissions_cdr[year] = np.NaN

    emissions_cdr.set_index(
        [
            "model",
            "scenario",
            "region",
            "sector",
            "product_category",
            "product_long",
            "product_short",
            "flow_category",
            "flow_long",
            "flow_short",
            "unit",
        ],
        inplace=True,
    )

    # change start year column to int if value is not NaN
    emissions_cdr["start year"] = (
        emissions_cdr["start year"].fillna(data_end_year + 1).astype(int)
    )

    # filter for rows where status is "Planned", "Under Construction", "Operational"
    emissions_cdr = emissions_cdr.loc[
        emissions_cdr.reset_index()
        .status.isin(["Planned", "Under Construction", "Operational"])
        .values
    ]

    # filter for rows where technology is "Direct Air Capture"
    emissions_cdr = emissions_cdr.loc[
        emissions_cdr.reset_index()
        .technology.isin(["Direct Air Capture"])
        .values
    ]

    # fill nans for Mt low and Mt high with median of Mt low and Mt high
    emissions_cdr["Mt low"] = emissions_cdr["Mt low"].fillna(
        emissions_cdr["Mt low"].median()
    )
    emissions_cdr["Mt high"] = emissions_cdr["Mt high"].fillna(
        emissions_cdr["Mt high"].median()
    )

    # for each row, put the median of the Mt low and Mt high values in the year column at the start year
    emissions_cdr = emissions_cdr.apply(
        lambda row: row.loc[row["start year"] : row["start year"]].fillna(
            row[["Mt low", "Mt high"]].median()
        )
        if row["start year"] >= data_start_year
        else row.loc[data_start_year:data_start_year].fillna(
            row[["Mt low", "Mt high"]].median()
        ),
        axis=1,
    )

    # Change region names to IEA
    regions = (
        pd.DataFrame(
            pd.read_csv(
                "podi/data/region_categories.csv",
                usecols=["WEB Region", "IEA_CCUS Region"],
            ).dropna(axis=0)
        )
        .set_index(["IEA_CCUS Region"])
        .rename_axis(index={"IEA_CCUS Region": "region"})
    )
    regions["WEB Region"] = (regions["WEB Region"]).str.lower()

    emissions_cdr = (
        (
            emissions_cdr.reset_index()
            .set_index(["region"])
            .merge(regions, on=["region"])
        )
        .reset_index()
        .set_index(
            [
                "model",
                "scenario",
                "WEB Region",
                "sector",
                "product_category",
                "product_long",
                "product_short",
                "flow_category",
                "flow_long",
                "flow_short",
                "unit",
            ]
        )
        .rename_axis(index={"WEB Region": "region"})
    ).drop(columns=["region"])

    # estimate cdr needed using difference between 2100 ppm and target (e.g. 350 ppm) using 7.8GtCo2e per ppm

    # fit adoption curve to cdr needed

    # allocate across regions and sectors proportional to current CDR activity

    # combine with emissions_output and emissions_output_co2e
    # emissions_output = pd.concat(emissions_output, emissions_cdr)

    # emissions_output_co2e = pd.concat(emissions_output_co2e, emissions_cdr)

    # # save output

    # emissions_output.columns = emissions_output.columns.astype(str)
    # for col in emissions_output.select_dtypes(include="float64").columns:
    #     emissions_output[col] = emissions_output[col].astype("float32")
    # emissions_output.sort_index().to_parquet(
    #     "podi/data/emissions_output.parquet", compression="brotli"
    # )

    # emissions_output_co2e.columns = emissions_output_co2e.columns.astype(str)
    # for col in emissions_output_co2e.select_dtypes(include="float64").columns:
    #     emissions_output_co2e[col] = emissions_output_co2e[col].astype(
    #         "float32"
    #     )
    # emissions_output_co2e.sort_index().to_parquet(
    #     "podi/data/emissions_output_co2e.parquet", compression="brotli"
    # )

    # endregion

    ###################
    # PANDAS DF AGENT #
    ###################
    """
    # region

    # set open_ai_key as environment variable
    os.environ["OPENAI_API_KEY"] = ""

    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0),
        emissions_output_co2e_biochar.T,
        verbose=True,
        chain_type="refine",
    )

    agent.run("how much co2 can biochar mitigate in 2030?")

    # endregion
    """
    ###########################################
    #  COMPARE OBSERVED TO MODELED EMISSIONS  #
    ###########################################

    # region

    # Load historical emissions data from ClimateTRACE
    emissions_historical = pd.concat(
        [
            pd.read_csv(
                "podi/data/ClimateTRACE/climatetrace_emissions_by_subsector_timeseries_interval_year_since_2015_to_2020.csv",
                usecols=[
                    "Tonnes Co2e",
                    "country",
                    "sector",
                    "subsector",
                    "start",
                ],
            ),
            pd.read_csv(
                "podi/data/ClimateTRACE/climatetrace_emissions_by_subsector_timeseries_sector_forests_since_2015_to_2020_interval_year.csv",
                usecols=[
                    "Tonnes Co2e",
                    "country",
                    "sector",
                    "subsector",
                    "start",
                ],
            ),
        ]
    )

    # Change ISO region names to IEA
    regions = (
        pd.DataFrame(
            pd.read_csv(
                "podi/data/region_categories.csv",
                usecols=["WEB Region", "ISO"],
            ).dropna(axis=0)
        )
        .set_index(["ISO"])
        .rename_axis(index={"ISO": "country"})
    )
    regions["WEB Region"] = (regions["WEB Region"]).str.lower()

    # Add model, scenario, and flow_category indices
    emissions_historical["model"] = "PD22"
    emissions_historical["scenario"] = "baseline"
    emissions_historical["product_category"] = "Emissions"
    emissions_historical["product_short"] = "EM"
    emissions_historical["flow_category"] = "Emissions"
    emissions_historical["flow_long"] = "Emissions"
    emissions_historical["flow_short"] = "EM"
    emissions_historical["unit"] = "MtCO2e"

    # Change unit from t to Mt
    emissions_historical["Tonnes Co2e"] = emissions_historical[
        "Tonnes Co2e"
    ].divide(1e6)

    # Change 'sector' index to 'product_long' and 'subsector' to 'flow_long' and
    # 'start' to 'year'
    emissions_historical.rename(
        columns={
            "Tonnes Co2e": "value",
            "subsector": "product_long",
            "start": "year",
        },
        inplace=True,
    )

    # Change 'year' format
    emissions_historical["year"] = (
        emissions_historical["year"]
        .str.split("-", expand=True)[0]
        .values.astype(int)
    )

    # Update Sector index
    def addsector4(x):
        if x["sector"] in ["power"]:
            return "Electric Power"
        elif x["sector"] in ["transport", "maritime"]:
            return "Transportation"
        elif x["sector"] in ["buildings"]:
            return "Buildings"
        elif x["sector"] in [
            "extraction",
            "manufacturing",
            "oil and gas",
            "waste",
        ]:
            return "Industrial"
        elif x["sector"] in ["agriculture"]:
            return "Agriculture"
        elif x["sector"] in ["forests"]:
            return "Forests & Wetlands"

    emissions_historical["sector"] = emissions_historical.parallel_apply(
        lambda x: addsector4(x), axis=1
    )

    emissions_historical = (
        (
            emissions_historical.reset_index()
            .set_index(["country"])
            .merge(regions, on=["country"])
        )
        .reset_index()
        .set_index(
            [
                "model",
                "scenario",
                "WEB Region",
                "sector",
                "product_category",
                "product_long",
                "product_short",
                "flow_category",
                "flow_long",
                "flow_short",
                "unit",
            ]
        )
        .rename_axis(index={"WEB Region": "region"})
    ).drop(columns=["country", "index"])

    # Pivot from long to wide
    emissions_historical = emissions_historical.reset_index().pivot(
        index=[
            "model",
            "scenario",
            "region",
            "sector",
            "product_category",
            "product_long",
            "product_short",
            "flow_category",
            "flow_long",
            "flow_short",
            "unit",
        ],
        columns="year",
        values="value",
    )

    # Select data between data_start_year and data_end_year
    emissions_historical.columns = emissions_historical.columns.astype(int)
    emissions_historical = emissions_historical.loc[
        :, data_start_year:data_end_year
    ]

    # Match modeled (emissions_output_co2e) and observed emissions
    # (emissions_historical) categories across 'model', 'region', 'sector'

    emissions_output_co2e_compare = (
        emissions_output_co2e.rename(
            index={"Residential": "Buildings", "Commercial": "Buildings"}
        )
        .groupby(["model", "region", "sector"], observed=True)
        .sum(numeric_only=True)
    )
    emissions_historical_compare = emissions_historical.groupby(
        ["model", "region", "sector"], observed=True
    ).sum(numeric_only=True)

    # Calculate error between modeled and observed
    emissions_error = abs(
        (
            emissions_historical_compare
            - emissions_output_co2e_compare.loc[
                :, emissions_historical_compare.columns
            ]
        )
        / emissions_historical_compare.loc[
            :, emissions_historical_compare.columns
        ]
    )

    # Drop observed emissions that are all zero
    emissions_error.replace([np.inf, -np.inf], np.nan, inplace=True)
    emissions_error.dropna(how="all", inplace=True)

    # endregion

    return
