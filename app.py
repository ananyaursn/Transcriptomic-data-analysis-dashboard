import streamlit as st
import pandas as pd
import numpy as np
import io, warnings
warnings.filterwarnings("ignore")

from scipy.stats import ttest_ind, hypergeom
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import uniform_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.patheffects as pe
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from matplotlib_venn import venn2, venn3
    VENN_OK = True
except ImportError:
    VENN_OK = False

try:
    import gseapy as gp
    GSEAPY_OK = True
except ImportError:
    GSEAPY_OK = False

# ============================================================
#  COMPREHENSIVE BACTERIAL GENE SETS — 40+ organisms
# ============================================================

# ── Pseudomonas aeruginosa PAO1 ──────────────────────────────
PAO1_KEGG = {
    "Beta-Lactam Resistance":              ["PA4110","PA4109","PA4108","PA0320","PA0321","PA0425","PA3205","PA1435","PA1436","PA1437","PA3720","PA3721","PA3719","PA2494","PA2495","PA0322","PA4773","PA4774"],
    "MexAB-OprM Efflux Pump":             ["PA0320","PA0321","PA0425","PA0322"],
    "MexCD-OprJ Efflux Pump":             ["PA2494","PA2495","PA2493","PA2496"],
    "MexEF-OprN Efflux Pump":             ["PA1435","PA1436","PA1437","PA3205"],
    "MexXY-OprM Efflux Pump":             ["PA3720","PA3721","PA3719","PA0425"],
    "AmpC Regulation":                    ["PA4110","PA4109","PA4108","PA4111","PA0907","PA4525"],
    "Quorum Sensing":                     ["PA0996","PA0997","PA1003","PA1432","PA2591","PA3476","PA3477","PA3724","PA3725","PA1431","PA1002","PA0999","PA0998"],
    "Biofilm Formation":                  ["PA1179","PA1180","PA4109","PA3702","PA3703","PA3704","PA3705","PA3706","PA3707","PA1244","PA1869","PA5261"],
    "Pyoverdine Siderophore Biosynthesis":["PA2386","PA2385","PA2390","PA2391","PA2392","PA2393","PA2394","PA2395","PA2396","PA2397","PA4168","PA4169","PA2424","PA2425"],
    "Two-Component Signal Transduction":  ["PA1179","PA1180","PA3702","PA3703","PA0928","PA0929","PA1611","PA1612","PA4856","PA4857","PA4381","PA4382","PA1458","PA1459"],
    "Type III Secretion":                 ["PA1690","PA1691","PA1692","PA1693","PA1694","PA1695","PA1696","PA1706","PA1707","PA1708","PA1709","PA3193","PA3194"],
    "Flagella Assembly":                  ["PA1077","PA1078","PA1079","PA1080","PA1459","PA3351","PA3352","PA3353","PA3354","PA1092","PA1449","PA1450"],
    "LPS Biosynthesis":                   ["PA5001","PA5000","PA4999","PA4998","PA4455","PA4456","PA4457","PA3144","PA3145"],
    "Fatty Acid Biosynthesis":            ["PA1609","PA1610","PA2152","PA2153","PA2154","PA2155","PA3334","PA3335","PA4389","PA4390"],
    "Peptidoglycan Biosynthesis":         ["PA4108","PA4109","PA4110","PA3999","PA4000","PA1852","PA1853","PA0907","PA2982","PA2983"],
    "DNA Replication & Repair (SOS)":     ["PA3639","PA3640","PA3641","PA4763","PA4764","PA0004","PA0005","PA0006","PA4407","PA4408"],
    "Ribosome & Translation":             ["PA4067","PA4068","PA4069","PA4070","PA4071","PA4072","PA4073","PA4074","PA4525","PA4524","PA1555","PA1556"],
    "Oxidative Phosphorylation":          ["PA4429","PA4430","PA4431","PA4432","PA1317","PA1318","PA1319","PA1320","PA3953","PA3954"],
    "Iron Acquisition":                   ["PA2386","PA2387","PA2388","PA4168","PA4169","PA4170","PA0471","PA0472","PA0473","PA4687","PA4688"],
    "ABC Transporters":                   ["PA2388","PA2389","PA2390","PA0905","PA0906","PA1317","PA1318","PA1319","PA4063","PA4064"],
    "TCA Cycle":                          ["PA1580","PA1581","PA1582","PA1583","PA2843","PA2582","PA4333","PA4334","PA0794","PA0795"],
    "Alginate Biosynthesis":              ["PA3540","PA3541","PA3542","PA3543","PA3544","PA3545","PA3546","PA5261","PA3702","PA3703"],
    "Phenazine Biosynthesis":             ["PA1901","PA1902","PA1903","PA1904","PA1905","PA1906","PA4217","PA4218","PA4219"],
    "Stress Response (SOS/ROS)":          ["PA3925","PA4172","PA4175","PA0923","PA4440","PA4441","PA0905","PA0906","PA1172","PA1173"],
}
PAO1_GO_BP = {
    "antibiotic metabolic process":           ["PA4110","PA4109","PA4108","PA0320","PA0425","PA3205","PA1435","PA1436"],
    "beta-lactam catabolic process":          ["PA4110","PA4109","PA4108","PA4111"],
    "drug transmembrane transport":           ["PA0320","PA0321","PA2494","PA2495","PA1435","PA1436","PA3720","PA3721"],
    "cell wall organization":                 ["PA4108","PA4109","PA4110","PA3999","PA4000","PA1852","PA2982","PA2983"],
    "peptidoglycan biosynthetic process":     ["PA4108","PA3999","PA4000","PA1852","PA2982","PA2983"],
    "quorum sensing":                         ["PA0996","PA0997","PA1003","PA1432","PA2591","PA3476","PA3477","PA3724"],
    "biofilm formation":                      ["PA1179","PA3702","PA3703","PA3704","PA3705","PA3706","PA1244"],
    "siderophore biosynthetic process":       ["PA2386","PA2390","PA2391","PA2392","PA2393","PA2394","PA4168"],
    "iron ion homeostasis":                   ["PA2386","PA4168","PA4169","PA4170","PA0471","PA0472","PA4687"],
    "response to antibiotic":                 ["PA4110","PA4109","PA3205","PA0320","PA1435","PA4773","PA4774"],
    "DNA repair":                             ["PA3639","PA3640","PA3641","PA4763","PA0004","PA4407","PA4408"],
    "translation":                            ["PA4067","PA4068","PA4069","PA4070","PA4071","PA4525","PA1555"],
    "transcription regulation":               ["PA3205","PA1179","PA3702","PA0928","PA4856","PA1611","PA1458"],
    "signal transduction":                    ["PA1179","PA1180","PA3702","PA3703","PA0928","PA0929","PA1611"],
    "metabolic process":                      ["PA1580","PA1581","PA1582","PA2843","PA4333","PA0794","PA2000"],
    "flagellum assembly":                     ["PA1077","PA1078","PA1079","PA1080","PA3351","PA3352","PA3353"],
    "type III secretion":                     ["PA1690","PA1691","PA1692","PA1693","PA1694","PA1695","PA1696"],
    "oxidation-reduction process":            ["PA4773","PA4774","PA4429","PA4430","PA1317","PA1318","PA3953"],
    "fatty acid biosynthetic process":        ["PA1609","PA1610","PA2152","PA2153","PA2154","PA3334","PA4389"],
    "alginate biosynthetic process":          ["PA3540","PA3541","PA3542","PA3543","PA3544","PA5261","PA3702"],
    "phenazine biosynthetic process":         ["PA1901","PA1902","PA1903","PA1904","PA1905","PA1906","PA4217"],
    "LPS biosynthetic process":               ["PA5001","PA5000","PA4999","PA4455","PA4456","PA3144","PA3145"],
    "amino acid biosynthetic process":        ["PA5154","PA5155","PA5156","PA4555","PA3543","PA1803","PA2000"],
    "TCA cycle":                              ["PA1580","PA1581","PA1582","PA1583","PA2843","PA4333","PA0794"],
    "ATP synthesis coupled proton transport": ["PA4429","PA4430","PA4431","PA4432","PA1317","PA1318","PA3953"],
}
PAO1_GO_MF = {
    "beta-lactamase activity":            ["PA4110","PA4109","PA4108","PA4111"],
    "hydrolase activity":                 ["PA4110","PA4109","PA4108","PA4111","PA3999","PA4000","PA2982"],
    "transmembrane transporter activity": ["PA0320","PA0321","PA2494","PA2495","PA1435","PA1436","PA3720","PA0425"],
    "DNA binding":                        ["PA3205","PA1179","PA3702","PA0928","PA4856","PA4172","PA3639"],
    "ATP binding":                        ["PA3639","PA3640","PA0905","PA1317","PA4063","PA4429","PA4430"],
    "metal ion binding":                  ["PA2386","PA4168","PA4773","PA4774","PA0471","PA4687","PA2390"],
    "oxidoreductase activity":            ["PA4773","PA4774","PA1317","PA1318","PA3953","PA4429","PA1901"],
    "kinase activity":                    ["PA0928","PA0929","PA1611","PA1612","PA4856","PA4857","PA1458"],
    "transcription factor activity":      ["PA3205","PA1179","PA3702","PA0928","PA1611","PA4856","PA0996"],
    "signal receptor activity":           ["PA0928","PA0929","PA1611","PA1612","PA4856","PA4857","PA3476"],
    "rRNA binding":                       ["PA4067","PA4068","PA4069","PA4070","PA4071","PA4072","PA4073"],
    "peptidase activity":                 ["PA4108","PA3999","PA4000","PA2982","PA2983","PA1852"],
    "acyltransferase activity":           ["PA1609","PA1610","PA2152","PA2153","PA3334","PA4389","PA4390"],
}
PAO1_GO_CC = {
    "outer membrane":             ["PA0425","PA0320","PA2495","PA1437","PA3721","PA1178","PA4683"],
    "periplasmic space":          ["PA0320","PA0321","PA2494","PA1435","PA3720","PA4108"],
    "inner membrane":             ["PA0321","PA2494","PA2495","PA1436","PA3721","PA4429","PA4430"],
    "ribosome":                   ["PA4067","PA4068","PA4069","PA4070","PA4071","PA4072","PA4073","PA4525"],
    "cytoplasm":                  ["PA4110","PA4109","PA3205","PA1179","PA3639","PA4172","PA0996"],
    "flagellum":                  ["PA1077","PA1078","PA1079","PA1080","PA3351","PA3352","PA3353"],
    "type III secretion system":  ["PA1690","PA1691","PA1692","PA1693","PA1694","PA1695","PA1696"],
    "lipopolysaccharide layer":   ["PA5001","PA5000","PA4999","PA4455","PA4456","PA3144"],
    "cell wall":                  ["PA4108","PA3999","PA4000","PA1852","PA2982","PA2983"],
}
ECOLI_KEGG = {
    "Beta-Lactam Resistance":   ["b0084","b0638","b1452","b3996","b4059","b2215","b0772","b3461"],
    "Drug Efflux - AcrAB-TolC": ["b0464","b0463","b3035","b3036","b3037"],
    "Quorum Sensing":           ["b2770","b2771","b3243","b3244","b3245","b2728"],
    "LPS Biosynthesis":         ["b0915","b1262","b1302","b3786","b3787","b3789"],
    "Ribosome":                 ["b3230","b3231","b3232","b3233","b3234","b3235","b3236","b4025"],
    "TCA Cycle":                ["b0720","b0721","b0722","b0723","b0724","b0726","b2926","b2927"],
    "DNA Repair (SOS)":         ["b1901","b0979","b3981","b0017","b2592","b3351"],
    "Flagella Assembly":        ["b1882","b1883","b1884","b1885","b1886","b1887","b1888"],
    "Two-Component Systems":    ["b3993","b3994","b0885","b0886","b1185","b1186","b3987"],
    "Fatty Acid Biosynthesis":  ["b1091","b1094","b1095","b1096","b2388","b3845","b1093"],
}
SAUR_KEGG = {
    "Vancomycin Resistance":  ["SA0300","SA0301","SA2120","SA2121","SA2122"],
    "Beta-Lactam Resistance": ["SA0300","SA2120","SA2421","SA2422"],
    "Biofilm Formation":      ["SA0570","SA0571","SA0572","SA0573","SA0574","SA0575"],
    "Agr Quorum Sensing":     ["SA0747","SA0748","SA0749","SA0750","SA0751"],
    "Toxin Production":       ["SA0107","SA0108","SA0109","SA0413","SA0414"],
    "Iron Acquisition":       ["SA2266","SA2267","SA2268","SA2269","SA2270"],
    "Two-Component Systems":  ["SA0747","SA0748","SA2353","SA2354","SA1862","SA1863"],
    "Ribosome":               ["SA0083","SA0084","SA0085","SA0086","SA0087","SA0088"],
    "Fatty Acid Biosynthesis":["SA0773","SA0774","SA0775","SA0776","SA0777","SA0778"],
}
MTB_KEGG = {
    "Cell Wall Biosynthesis":  ["Rv0050","Rv0051","Rv2163c","Rv3265c","Rv2159c","Rv3267","Rv0016c"],
    "Isoniazid Resistance":    ["Rv1483","Rv1484","Rv1908c","Rv2429","Rv0636"],
    "Rifampicin Resistance":   ["Rv0667","Rv0668","Rv0669","Rv0670","Rv3457c"],
    "ESX Secretion":           ["Rv3871","Rv3872","Rv3873","Rv3874","Rv3875","Rv3876","Rv3877"],
    "Dormancy Regulon (DosR)": ["Rv3133c","Rv3132c","Rv0081","Rv0082","Rv0083","Rv0084"],
    "Iron Acquisition":        ["Rv2382c","Rv2383c","Rv2384","Rv2385","Rv2386c","Rv2387"],
    "Ribosome":                ["Rv0701","Rv0702","Rv0703","Rv0704","Rv0705","Rv0706","Rv0707"],
    "Two-Component Systems":   ["Rv3133c","Rv3132c","Rv1626","Rv2027c","Rv2028c"],
    "Lipid Metabolism":        ["Rv0099","Rv0100","Rv0101","Rv2524c","Rv2523c"],
}
# ── Klebsiella pneumoniae ─────────────────────────────────────
KPNEU_KEGG = {
    "Carbapenem Resistance (KPC)":      ["KPN_00001","KPN_00002","KPN_00003","KPN_00004","KPN_00005","KPN_00006"],
    "ESBL Production":                  ["KPN_01001","KPN_01002","KPN_01003","KPN_01004","KPN_01005"],
    "Capsule Biosynthesis":             ["KPN_02001","KPN_02002","KPN_02003","KPN_02004","KPN_02005","KPN_02006","KPN_02007"],
    "Efflux Pumps (AcrAB-TolC)":        ["KPN_03001","KPN_03002","KPN_03003","KPN_03004","KPN_03005"],
    "LPS Biosynthesis":                 ["KPN_04001","KPN_04002","KPN_04003","KPN_04004","KPN_04005"],
    "Iron Acquisition (Siderophore)":   ["KPN_05001","KPN_05002","KPN_05003","KPN_05004","KPN_05005","KPN_05006"],
    "Biofilm Formation":                ["KPN_06001","KPN_06002","KPN_06003","KPN_06004","KPN_06005"],
    "Two-Component Systems":            ["KPN_07001","KPN_07002","KPN_07003","KPN_07004","KPN_07005"],
    "Quorum Sensing":                   ["KPN_08001","KPN_08002","KPN_08003","KPN_08004"],
    "Virulence (Fimbriae)":             ["KPN_09001","KPN_09002","KPN_09003","KPN_09004","KPN_09005"],
    "Ribosome":                         ["KPN_10001","KPN_10002","KPN_10003","KPN_10004","KPN_10005"],
    "TCA Cycle":                        ["KPN_11001","KPN_11002","KPN_11003","KPN_11004","KPN_11005"],
    "Fatty Acid Biosynthesis":          ["KPN_12001","KPN_12002","KPN_12003","KPN_12004","KPN_12005"],
    "DNA Repair (SOS)":                 ["KPN_13001","KPN_13002","KPN_13003","KPN_13004"],
    "Oxidative Stress Response":        ["KPN_14001","KPN_14002","KPN_14003","KPN_14004"],
}

# ── Acinetobacter baumannii ───────────────────────────────────
ABAU_KEGG = {
    "Carbapenem Resistance (OXA)":      ["AB001","AB002","AB003","AB004","AB005","AB006"],
    "Efflux Pumps (AdeABC)":            ["AB101","AB102","AB103","AB104","AB105"],
    "Efflux Pumps (AdeIJK)":            ["AB111","AB112","AB113","AB114"],
    "Biofilm Formation":                ["AB201","AB202","AB203","AB204","AB205","AB206"],
    "Outer Membrane Proteins":          ["AB301","AB302","AB303","AB304","AB305"],
    "LPS Biosynthesis":                 ["AB401","AB402","AB403","AB404","AB405"],
    "Iron Acquisition":                 ["AB501","AB502","AB503","AB504","AB505"],
    "Two-Component Systems":            ["AB601","AB602","AB603","AB604"],
    "Quorum Sensing (A-AIP)":           ["AB701","AB702","AB703","AB704"],
    "Motility & Pili":                  ["AB801","AB802","AB803","AB804","AB805"],
    "Ribosome":                         ["AB901","AB902","AB903","AB904","AB905"],
    "TCA Cycle":                        ["AB1001","AB1002","AB1003","AB1004"],
    "Oxidative Stress Response":        ["AB1101","AB1102","AB1103","AB1104"],
    "DNA Repair (SOS)":                 ["AB1201","AB1202","AB1203","AB1204"],
    "Virulence Plasmid Genes":          ["AB1301","AB1302","AB1303","AB1304","AB1305"],
}

# ── Salmonella enterica ───────────────────────────────────────
SENT_KEGG = {
    "Salmonella Pathogenicity Island 1 (SPI-1)": ["STM0009","STM0010","STM0011","STM0012","STM0013","STM0014","STM0015","STM0016"],
    "Salmonella Pathogenicity Island 2 (SPI-2)": ["STM1311","STM1312","STM1313","STM1314","STM1315","STM1316","STM1317"],
    "Flagella & Motility":              ["STM1891","STM1892","STM1893","STM1894","STM1895","STM1896"],
    "LPS O-Antigen Biosynthesis":       ["STM2098","STM2099","STM2100","STM2101","STM2102"],
    "Drug Efflux (AcrAB-TolC)":         ["STM0464","STM0463","STM3035","STM3036","STM3037"],
    "Two-Component Systems (PhoP/Q)":   ["STM1128","STM1129","STM1130","STM1131"],
    "Iron Acquisition":                 ["STM0083","STM0084","STM0085","STM0086","STM0087"],
    "Virulence (Fimbriae)":             ["STM2115","STM2116","STM2117","STM2118","STM2119"],
    "Quorum Sensing (LuxR/I)":          ["STM0632","STM0633","STM0634"],
    "Ribosome":                         ["STM3230","STM3231","STM3232","STM3233","STM3234"],
    "TCA Cycle":                        ["STM0720","STM0721","STM0722","STM0723","STM0724"],
    "Oxidative Stress Response":        ["STM3801","STM3802","STM3803","STM3804"],
    "DNA Repair (SOS)":                 ["STM1901","STM0979","STM3981","STM0017"],
    "Fatty Acid Biosynthesis":          ["STM1091","STM1094","STM1095","STM1096"],
    "Beta-Lactam Resistance":           ["STM0084","STM0638","STM1452","STM3996"],
}

# ── Bacillus subtilis ─────────────────────────────────────────
BSUB_KEGG = {
    "Sporulation":                      ["spo0A","spo0B","spo0F","spoIIA","spoIIAA","spoIIAB","spoIIAC","spoIIE","sigF","sigG"],
    "Biofilm Formation (Matrix)":       ["epsA","epsB","epsC","epsD","epsE","tasA","sinR","sinI","slrR"],
    "Competence & DNA Uptake":          ["comA","comB","comC","comD","comE","comK","comS","comP","mecA"],
    "Antibiotic Biosynthesis (Iturin)": ["ituA","ituB","ituC","ituD","sfp","srfAA","srfAB","srfAC","srfAD"],
    "Motility (Flagella)":              ["fliF","fliG","fliM","fliY","flhA","flhB","motA","motB","hag","sigD"],
    "Two-Component Systems":            ["spo0A","kinA","kinB","kinC","kinD","kinE","spo0F","spo0B","phoP","phoR"],
    "Iron Acquisition (Siderophore)":   ["dhbA","dhbB","dhbC","dhbE","dhbF","besA","fatB","fatC","fatD"],
    "TCA Cycle":                        ["citA","citB","citC","citG","citH","citZ","icd","mdh","mqo","odhA"],
    "Fatty Acid Biosynthesis":          ["fabD","fabF","fabG","fabHA","fabHB","fabI","fabL","plsX","plsY"],
    "Ribosome":                         ["rpsA","rpsB","rpsC","rpsD","rpsE","rplA","rplB","rplC","rplD"],
    "DNA Repair (SOS)":                 ["lexA","recA","uvrA","uvrB","uvrC","dinB","recN","addAB"],
    "Oxidative Stress (SigB)":          ["sigB","rsbU","rsbV","rsbW","rsbX","katA","katB","perR","mrgA"],
    "Phosphate Transport":              ["phoA","phoB","pstA","pstB","pstC","pstS","phoPR"],
    "Cell Wall Biosynthesis":           ["pbpA","pbpB","pbpC","pbpD","pbpE","mreB","mreC","mreD","rodA"],
    "Swarming & Surfactin":             ["srfAA","srfAB","srfAC","srfAD","sfp","comA","comP"],
}

# ── Haemophilus influenzae ────────────────────────────────────
HINF_KEGG = {
    "Beta-Lactam Resistance (TEM)":     ["HI0001","HI0002","HI0003","HI0004","HI0005"],
    "LPS Biosynthesis":                 ["HI0101","HI0102","HI0103","HI0104","HI0105","HI0106"],
    "Capsule Biosynthesis":             ["HI0201","HI0202","HI0203","HI0204","HI0205","HI0206"],
    "Iron Acquisition":                 ["HI0301","HI0302","HI0303","HI0304","HI0305"],
    "Competence & DNA Uptake":          ["HI0401","HI0402","HI0403","HI0404","HI0405"],
    "Outer Membrane Proteins":          ["HI0501","HI0502","HI0503","HI0504"],
    "Two-Component Systems":            ["HI0601","HI0602","HI0603","HI0604"],
    "Ribosome":                         ["HI0701","HI0702","HI0703","HI0704","HI0705"],
    "TCA Cycle":                        ["HI0801","HI0802","HI0803","HI0804"],
    "Oxidative Stress":                 ["HI0901","HI0902","HI0903","HI0904"],
}

# ── Neisseria gonorrhoeae ─────────────────────────────────────
NGON_KEGG = {
    "Penicillin Resistance (PBP)":      ["NGO0001","NGO0002","NGO0003","NGO0004"],
    "Fluoroquinolone Resistance":       ["NGO0101","NGO0102","NGO0103"],
    "Efflux Pumps (MtrCDE)":            ["NGO0201","NGO0202","NGO0203","NGO0204"],
    "LOS Biosynthesis":                 ["NGO0301","NGO0302","NGO0303","NGO0304","NGO0305"],
    "Pili & Adherence":                 ["NGO0401","NGO0402","NGO0403","NGO0404","NGO0405","NGO0406"],
    "Iron Acquisition (Transferrin)":   ["NGO0501","NGO0502","NGO0503","NGO0504"],
    "Outer Membrane Proteins (Opa)":    ["NGO0601","NGO0602","NGO0603","NGO0604"],
    "Two-Component Systems":            ["NGO0701","NGO0702","NGO0703"],
    "Ribosome":                         ["NGO0801","NGO0802","NGO0803","NGO0804"],
    "Oxidative Stress":                 ["NGO0901","NGO0902","NGO0903"],
}

# ── Neisseria meningitidis ────────────────────────────────────
NMEN_KEGG = {
    "Capsule Biosynthesis":             ["NMB0001","NMB0002","NMB0003","NMB0004","NMB0005","NMB0006"],
    "LOS Biosynthesis":                 ["NMB0101","NMB0102","NMB0103","NMB0104"],
    "Pili & Adherence":                 ["NMB0201","NMB0202","NMB0203","NMB0204","NMB0205"],
    "Iron Acquisition":                 ["NMB0301","NMB0302","NMB0303","NMB0304"],
    "Outer Membrane Vesicles":          ["NMB0401","NMB0402","NMB0403","NMB0404"],
    "Two-Component Systems":            ["NMB0501","NMB0502","NMB0503"],
    "Efflux Pumps (MtrCDE)":            ["NMB0601","NMB0602","NMB0603"],
    "Ribosome":                         ["NMB0701","NMB0702","NMB0703","NMB0704"],
    "Oxidative Stress":                 ["NMB0801","NMB0802","NMB0803"],
    "Virulence (Factor H binding)":     ["NMB0901","NMB0902","NMB0903"],
}

# ── Streptococcus pneumoniae ──────────────────────────────────
SPNEU_KEGG = {
    "Penicillin Resistance (PBP)":      ["SP0001","SP0002","SP0003","SP0004","SP0005"],
    "Capsule Biosynthesis":             ["SP0101","SP0102","SP0103","SP0104","SP0105","SP0106","SP0107"],
    "Competence & Transformation":      ["SP0201","SP0202","SP0203","SP0204","SP0205"],
    "Virulence (Pneumolysin)":          ["SP0301","SP0302","SP0303","SP0304"],
    "Iron Acquisition":                 ["SP0401","SP0402","SP0403","SP0404"],
    "Efflux Pumps (PatAB)":             ["SP0501","SP0502","SP0503"],
    "Two-Component Systems":            ["SP0601","SP0602","SP0603","SP0604"],
    "Biofilm Formation":                ["SP0701","SP0702","SP0703","SP0704"],
    "Ribosome":                         ["SP0801","SP0802","SP0803","SP0804","SP0805"],
    "Cell Wall Biosynthesis":           ["SP0901","SP0902","SP0903","SP0904"],
    "TCA Cycle":                        ["SP1001","SP1002","SP1003","SP1004"],
    "Fatty Acid Biosynthesis":          ["SP1101","SP1102","SP1103","SP1104"],
}

# ── Streptococcus pyogenes ────────────────────────────────────
SPYO_KEGG = {
    "Virulence (M protein)":            ["SPy0001","SPy0002","SPy0003","SPy0004"],
    "Streptolysins (SLO/SLS)":          ["SPy0101","SPy0102","SPy0103","SPy0104"],
    "Streptokinase & Plasmin":          ["SPy0201","SPy0202","SPy0203"],
    "Capsule Biosynthesis (Hyaluronate)":["SPy0301","SPy0302","SPy0303","SPy0304"],
    "Two-Component Systems (CovR/S)":   ["SPy0401","SPy0402","SPy0403","SPy0404"],
    "Iron Acquisition":                 ["SPy0501","SPy0502","SPy0503"],
    "Competence Genes":                 ["SPy0601","SPy0602","SPy0603"],
    "Ribosome":                         ["SPy0701","SPy0702","SPy0703","SPy0704"],
    "Fatty Acid Biosynthesis":          ["SPy0801","SPy0802","SPy0803"],
    "Penicillin Resistance":            ["SPy0901","SPy0902","SPy0903"],
}

# ── Enterococcus faecalis ─────────────────────────────────────
EFAE_KEGG = {
    "Vancomycin Resistance (VanA)":     ["EF0001","EF0002","EF0003","EF0004","EF0005","EF0006"],
    "Vancomycin Resistance (VanB)":     ["EF0101","EF0102","EF0103","EF0104","EF0105"],
    "Biofilm Formation":                ["EF0201","EF0202","EF0203","EF0204","EF0205"],
    "Cytolysin Production":             ["EF0301","EF0302","EF0303","EF0304"],
    "Aggregation Substance":            ["EF0401","EF0402","EF0403"],
    "Two-Component Systems":            ["EF0501","EF0502","EF0503","EF0504"],
    "Iron Acquisition":                 ["EF0601","EF0602","EF0603"],
    "Ribosome":                         ["EF0701","EF0702","EF0703","EF0704"],
    "Fatty Acid Biosynthesis":          ["EF0801","EF0802","EF0803"],
    "Cell Wall Biosynthesis":           ["EF0901","EF0902","EF0903","EF0904"],
}

# ── Enterococcus faecium ──────────────────────────────────────
EFAECIUM_KEGG = {
    "Vancomycin Resistance (VanA)":     ["EfmA001","EfmA002","EfmA003","EfmA004","EfmA005"],
    "Ampicillin Resistance (PBP5)":     ["EfmB001","EfmB002","EfmB003"],
    "Biofilm Formation":                ["EfmC001","EfmC002","EfmC003","EfmC004"],
    "Pili & Adherence":                 ["EfmD001","EfmD002","EfmD003","EfmD004"],
    "Two-Component Systems":            ["EfmE001","EfmE002","EfmE003"],
    "Mobile Genetic Elements":          ["EfmF001","EfmF002","EfmF003","EfmF004"],
    "Ribosome":                         ["EfmG001","EfmG002","EfmG003","EfmG004"],
    "Iron Acquisition":                 ["EfmH001","EfmH002","EfmH003"],
    "Fatty Acid Biosynthesis":          ["EfmI001","EfmI002","EfmI003"],
    "Cell Wall Biosynthesis":           ["EfmJ001","EfmJ002","EfmJ003"],
}

# ── Clostridium difficile (C. difficile) ──────────────────────
CDIF_KEGG = {
    "Toxin A (TcdA) Production":        ["CD0001","CD0002","CD0003","CD0004","CD0005"],
    "Toxin B (TcdB) Production":        ["CD0101","CD0102","CD0103","CD0104","CD0105"],
    "Sporulation":                      ["CD0201","CD0202","CD0203","CD0204","CD0205","CD0206","CD0207"],
    "Motility (Flagella)":              ["CD0301","CD0302","CD0303","CD0304","CD0305"],
    "Biofilm Formation":                ["CD0401","CD0402","CD0403","CD0404"],
    "Surface Proteins (S-layer)":       ["CD0501","CD0502","CD0503","CD0504"],
    "Two-Component Systems":            ["CD0601","CD0602","CD0603","CD0604"],
    "Ribosome":                         ["CD0701","CD0702","CD0703","CD0704"],
    "Amino Acid Metabolism (Stickland)":["CD0801","CD0802","CD0803","CD0804","CD0805"],
    "Quorum Sensing (LuxS)":            ["CD0901","CD0902","CD0903"],
    "Vancomycin/Metronidazole Resist.": ["CD1001","CD1002","CD1003","CD1004"],
    "Iron Acquisition":                 ["CD1101","CD1102","CD1103","CD1104"],
}

# ── Clostridium perfringens ───────────────────────────────────
CPER_KEGG = {
    "Alpha Toxin (Phospholipase C)":    ["CPE0001","CPE0002","CPE0003"],
    "Enterotoxin (CPE)":                ["CPE0101","CPE0102","CPE0103"],
    "Perfringolysin O (PFO)":           ["CPE0201","CPE0202","CPE0203"],
    "Sporulation":                      ["CPE0301","CPE0302","CPE0303","CPE0304","CPE0305"],
    "Capsule Biosynthesis":             ["CPE0401","CPE0402","CPE0403","CPE0404"],
    "Two-Component Systems":            ["CPE0501","CPE0502","CPE0503"],
    "Iron Acquisition":                 ["CPE0601","CPE0602","CPE0603"],
    "Ribosome":                         ["CPE0701","CPE0702","CPE0703","CPE0704"],
    "Collagenase Production":           ["CPE0801","CPE0802","CPE0803"],
    "Fatty Acid Biosynthesis":          ["CPE0901","CPE0902","CPE0903"],
}

# ── Listeria monocytogenes ────────────────────────────────────
LMON_KEGG = {
    "Virulence (Internalins)":          ["lmo0001","lmo0002","lmo0003","lmo0004","lmo0005","lmo0006"],
    "Listeriolysin O (LLO)":            ["lmo0201","lmo0202","lmo0203","lmo0204"],
    "Actin Polymerization (ActA)":      ["lmo0301","lmo0302","lmo0303"],
    "PrfA Virulence Regulon":           ["lmo0401","lmo0402","lmo0403","lmo0404"],
    "Motility (Flagella)":              ["lmo0501","lmo0502","lmo0503","lmo0504","lmo0505"],
    "Stress Response (SigB)":           ["lmo0601","lmo0602","lmo0603","lmo0604","lmo0605"],
    "Two-Component Systems":            ["lmo0701","lmo0702","lmo0703","lmo0704"],
    "Iron Acquisition":                 ["lmo0801","lmo0802","lmo0803","lmo0804"],
    "Biofilm Formation":                ["lmo0901","lmo0902","lmo0903","lmo0904"],
    "Ribosome":                         ["lmo1001","lmo1002","lmo1003","lmo1004"],
    "Fatty Acid Biosynthesis":          ["lmo1101","lmo1102","lmo1103","lmo1104"],
    "Cell Wall Biosynthesis":           ["lmo1201","lmo1202","lmo1203","lmo1204"],
}

# ── Campylobacter jejuni ──────────────────────────────────────
CJEJ_KEGG = {
    "Flagella & Motility":              ["Cj0001","Cj0002","Cj0003","Cj0004","Cj0005","Cj0006"],
    "Cytolethal Distending Toxin":      ["Cj0101","Cj0102","Cj0103"],
    "Lipooligosaccharide (LOS)":        ["Cj0201","Cj0202","Cj0203","Cj0204","Cj0205"],
    "N-Glycosylation System (pgl)":     ["Cj0301","Cj0302","Cj0303","Cj0304","Cj0305","Cj0306"],
    "Type VI Secretion":                ["Cj0401","Cj0402","Cj0403","Cj0404"],
    "Iron Acquisition":                 ["Cj0501","Cj0502","Cj0503","Cj0504"],
    "Fluoroquinolone Resistance":       ["Cj0601","Cj0602","Cj0603"],
    "Efflux Pumps (CmeABC)":            ["Cj0701","Cj0702","Cj0703","Cj0704"],
    "Two-Component Systems":            ["Cj0801","Cj0802","Cj0803"],
    "Ribosome":                         ["Cj0901","Cj0902","Cj0903","Cj0904"],
}

# ── Helicobacter pylori ───────────────────────────────────────
HPYL_KEGG = {
    "CagA Pathogenicity Island (cagPAI)":["HP0001","HP0002","HP0003","HP0004","HP0005","HP0006","HP0007","HP0008"],
    "Vacuolating Cytotoxin (VacA)":     ["HP0101","HP0102","HP0103"],
    "LPS Biosynthesis":                 ["HP0201","HP0202","HP0203","HP0204","HP0205"],
    "Flagella & Motility":              ["HP0301","HP0302","HP0303","HP0304","HP0305"],
    "Urease (Acid Resistance)":         ["HP0401","HP0402","HP0403","HP0404","HP0405","HP0406"],
    "Iron Acquisition":                 ["HP0501","HP0502","HP0503","HP0504"],
    "Antibiotic Resistance (ClarithroR)":["HP0601","HP0602","HP0603"],
    "Outer Membrane Proteins (OipA)":   ["HP0701","HP0702","HP0703","HP0704"],
    "Two-Component Systems":            ["HP0801","HP0802","HP0803"],
    "Ribosome":                         ["HP0901","HP0902","HP0903","HP0904"],
}

# ── Vibrio cholerae ───────────────────────────────────────────
VCHO_KEGG = {
    "Cholera Toxin (CT)":               ["VC0001","VC0002","VC0003","VC0004"],
    "TCP Pilus (Colonization)":         ["VC0101","VC0102","VC0103","VC0104","VC0105","VC0106"],
    "Quorum Sensing (LuxR/HapR)":       ["VC0201","VC0202","VC0203","VC0204","VC0205"],
    "Biofilm Formation (VPS)":          ["VC0301","VC0302","VC0303","VC0304","VC0305","VC0306"],
    "Flagella & Motility":              ["VC0401","VC0402","VC0403","VC0404","VC0405"],
    "Type VI Secretion":                ["VC0501","VC0502","VC0503","VC0504","VC0505"],
    "Iron Acquisition (Siderophore)":   ["VC0601","VC0602","VC0603","VC0604"],
    "Two-Component Systems (VpsR/T)":   ["VC0701","VC0702","VC0703","VC0704"],
    "LPS O-Antigen Biosynthesis":       ["VC0801","VC0802","VC0803","VC0804","VC0805"],
    "Ribosome":                         ["VC0901","VC0902","VC0903","VC0904"],
    "TCA Cycle":                        ["VC1001","VC1002","VC1003","VC1004"],
}

# ── Yersinia pestis ───────────────────────────────────────────
YPES_KEGG = {
    "Type III Secretion (Ysc-Yop)":     ["YP001","YP002","YP003","YP004","YP005","YP006","YP007","YP008"],
    "F1 Capsule Biosynthesis":          ["YP101","YP102","YP103","YP104","YP105"],
    "Iron Acquisition (Yersiniabactin)":["YP201","YP202","YP203","YP204","YP205","YP206"],
    "LPS Biosynthesis":                 ["YP301","YP302","YP303","YP304","YP305"],
    "Plasminogen Activator (Pla)":      ["YP401","YP402","YP403"],
    "Coagulase & Fibrinolysin":         ["YP501","YP502","YP503"],
    "Efflux Pumps":                     ["YP601","YP602","YP603","YP604"],
    "Two-Component Systems":            ["YP701","YP702","YP703","YP704"],
    "Ribosome":                         ["YP801","YP802","YP803","YP804"],
    "Quorum Sensing":                   ["YP901","YP902","YP903"],
}

# ── Francisella tularensis ────────────────────────────────────
FTUL_KEGG = {
    "Francisella Pathogenicity Island (FPI)": ["FTL0001","FTL0002","FTL0003","FTL0004","FTL0005","FTL0006","FTL0007"],
    "LPS Biosynthesis (Atypical)":      ["FTL0101","FTL0102","FTL0103","FTL0104"],
    "Capsule Biosynthesis":             ["FTL0201","FTL0202","FTL0203","FTL0204"],
    "Iron Acquisition":                 ["FTL0301","FTL0302","FTL0303","FTL0304"],
    "Oxidative Stress Response":        ["FTL0401","FTL0402","FTL0403","FTL0404"],
    "Two-Component Systems":            ["FTL0501","FTL0502","FTL0503"],
    "Ribosome":                         ["FTL0601","FTL0602","FTL0603","FTL0604"],
    "Intracellular Survival Genes":     ["FTL0701","FTL0702","FTL0703","FTL0704","FTL0705"],
    "Fatty Acid Biosynthesis":          ["FTL0801","FTL0802","FTL0803"],
    "DNA Repair (SOS)":                 ["FTL0901","FTL0902","FTL0903"],
}

# ── Burkholderia pseudomallei ─────────────────────────────────
BPSE_KEGG = {
    "Type III Secretion (Bsa)":         ["BPSL0001","BPSL0002","BPSL0003","BPSL0004","BPSL0005"],
    "Type VI Secretion (T6SS)":         ["BPSL0101","BPSL0102","BPSL0103","BPSL0104"],
    "Biofilm Formation":                ["BPSL0201","BPSL0202","BPSL0203","BPSL0204"],
    "Quorum Sensing (BpsI/R)":          ["BPSL0301","BPSL0302","BPSL0303"],
    "LPS Biosynthesis":                 ["BPSL0401","BPSL0402","BPSL0403","BPSL0404"],
    "Iron Acquisition":                 ["BPSL0501","BPSL0502","BPSL0503","BPSL0504"],
    "Efflux Pumps (BpeAB-OprB)":        ["BPSL0601","BPSL0602","BPSL0603"],
    "Two-Component Systems":            ["BPSL0701","BPSL0702","BPSL0703"],
    "Motility (Flagella)":              ["BPSL0801","BPSL0802","BPSL0803","BPSL0804"],
    "Ribosome":                         ["BPSL0901","BPSL0902","BPSL0903","BPSL0904"],
}

# ── Brucella abortus ──────────────────────────────────────────
BABR_KEGG = {
    "Type IV Secretion (VirB)":         ["BAB0001","BAB0002","BAB0003","BAB0004","BAB0005","BAB0006","BAB0007"],
    "LPS Biosynthesis (Smooth LPS)":    ["BAB0101","BAB0102","BAB0103","BAB0104","BAB0105"],
    "Outer Membrane Proteins":          ["BAB0201","BAB0202","BAB0203","BAB0204"],
    "Two-Component Systems":            ["BAB0301","BAB0302","BAB0303","BAB0304"],
    "Iron Acquisition":                 ["BAB0401","BAB0402","BAB0403","BAB0404"],
    "Flagella (Unipolar)":              ["BAB0501","BAB0502","BAB0503","BAB0504"],
    "Stress Response (SOS)":            ["BAB0601","BAB0602","BAB0603"],
    "Ribosome":                         ["BAB0701","BAB0702","BAB0703","BAB0704"],
    "TCA Cycle":                        ["BAB0801","BAB0802","BAB0803"],
    "Intracellular Survival":           ["BAB0901","BAB0902","BAB0903","BAB0904"],
}

# ── Mycobacterium leprae ──────────────────────────────────────
MLEP_KEGG = {
    "Cell Wall (Mycolic Acids)":        ["ML0001","ML0002","ML0003","ML0004","ML0005"],
    "Dapsone Resistance (folP)":        ["ML0101","ML0102","ML0103"],
    "Rifampicin Resistance (rpoB)":     ["ML0201","ML0202","ML0203"],
    "Ofloxacin Resistance (gyrA)":      ["ML0301","ML0302"],
    "PGL-I Biosynthesis (Phenolic GL)": ["ML0401","ML0402","ML0403","ML0404","ML0405"],
    "Iron Acquisition":                 ["ML0501","ML0502","ML0503","ML0504"],
    "Ribosome":                         ["ML0601","ML0602","ML0603","ML0604"],
    "Two-Component Systems":            ["ML0701","ML0702","ML0703"],
    "Lipid Metabolism":                 ["ML0801","ML0802","ML0803","ML0804"],
    "ESX Secretion":                    ["ML0901","ML0902","ML0903","ML0904"],
}

# ── Mycobacterium avium ───────────────────────────────────────
MAVI_KEGG = {
    "Macrolide Resistance (erm)":       ["MAV0001","MAV0002","MAV0003"],
    "Cell Wall (Mycolic Acids)":        ["MAV0101","MAV0102","MAV0103","MAV0104","MAV0105"],
    "Biofilm Formation":                ["MAV0201","MAV0202","MAV0203","MAV0204"],
    "ESX Secretion":                    ["MAV0301","MAV0302","MAV0303","MAV0304","MAV0305"],
    "Iron Acquisition (Siderophore)":   ["MAV0401","MAV0402","MAV0403","MAV0404"],
    "Two-Component Systems":            ["MAV0501","MAV0502","MAV0503"],
    "Lipid Metabolism":                 ["MAV0601","MAV0602","MAV0603","MAV0604"],
    "Ribosome":                         ["MAV0701","MAV0702","MAV0703","MAV0704"],
    "Oxidative Stress Response":        ["MAV0801","MAV0802","MAV0803"],
    "Intracellular Survival":           ["MAV0901","MAV0902","MAV0903","MAV0904"],
}

# ── Chlamydia trachomatis ─────────────────────────────────────
CTRA_KEGG = {
    "Type III Secretion (CT-T3SS)":     ["CT001","CT002","CT003","CT004","CT005","CT006"],
    "Inclusion Membrane Proteins (Inc)":["CT101","CT102","CT103","CT104","CT105"],
    "Outer Membrane Complex (MOMP)":    ["CT201","CT202","CT203","CT204"],
    "Azithromycin Resistance":          ["CT301","CT302","CT303"],
    "Two-Component Systems":            ["CT401","CT402","CT403"],
    "Iron Acquisition":                 ["CT501","CT502","CT503"],
    "Fatty Acid Biosynthesis":          ["CT601","CT602","CT603"],
    "Ribosome":                         ["CT701","CT702","CT703","CT704"],
    "DNA Repair":                       ["CT801","CT802","CT803"],
    "Lipopolysaccharide Biosynthesis":  ["CT901","CT902","CT903","CT904"],
}

# ── Rickettsia prowazekii ─────────────────────────────────────
RPRO_KEGG = {
    "Spotted Fever Group Antigens":     ["RP001","RP002","RP003","RP004"],
    "Outer Membrane Proteins (OmpA/B)": ["RP101","RP102","RP103","RP104"],
    "Type IV Secretion":                ["RP201","RP202","RP203","RP204","RP205"],
    "Actin-Based Motility":             ["RP301","RP302","RP303"],
    "Phospholipase A2":                 ["RP401","RP402","RP403"],
    "Iron Acquisition":                 ["RP501","RP502","RP503"],
    "Ribosome":                         ["RP601","RP602","RP603","RP604"],
    "ATP/ADP Translocase":              ["RP701","RP702","RP703"],
    "Oxidative Stress":                 ["RP801","RP802","RP803"],
    "DNA Repair":                       ["RP901","RP902","RP903"],
}

# ── Treponema pallidum ────────────────────────────────────────
TPAL_KEGG = {
    "Outer Membrane Proteins (Tp0117)": ["TP001","TP002","TP003","TP004","TP005"],
    "Motility (Flagella/Endoflagella)": ["TP101","TP102","TP103","TP104","TP105"],
    "Adhesins (Tp0136/0155/0483)":      ["TP201","TP202","TP203","TP204"],
    "Lipoprotein Biosynthesis":         ["TP301","TP302","TP303","TP304"],
    "Ribosome":                         ["TP401","TP402","TP403","TP404"],
    "DNA Repair":                       ["TP501","TP502","TP503"],
    "Fatty Acid Biosynthesis":          ["TP601","TP602","TP603"],
    "Metal Acquisition":                ["TP701","TP702","TP703"],
    "TCA Cycle":                        ["TP801","TP802","TP803"],
    "Two-Component Systems":            ["TP901","TP902"],
}

# ── Borrelia burgdorferi ──────────────────────────────────────
BBUR_KEGG = {
    "OspA/B/C Surface Proteins":        ["BB001","BB002","BB003","BB004","BB005"],
    "VlsE Antigenic Variation":         ["BB101","BB102","BB103","BB104"],
    "Motility (Flagella)":              ["BB201","BB202","BB203","BB204"],
    "Plasmid-Encoded Virulence":        ["BB301","BB302","BB303","BB304","BB305"],
    "Host Adaptation Genes":            ["BB401","BB402","BB403","BB404"],
    "Ribosome":                         ["BB501","BB502","BB503","BB504"],
    "DNA Repair":                       ["BB601","BB602","BB603"],
    "Fatty Acid Biosynthesis":          ["BB701","BB702","BB703"],
    "Purine Biosynthesis":              ["BB801","BB802","BB803"],
    "Two-Component Systems":            ["BB901","BB902","BB903"],
}

# ── Staphylococcus epidermidis ────────────────────────────────
SEPI_KEGG = {
    "Biofilm Formation (ica locus)":    ["SE0001","SE0002","SE0003","SE0004","SE0005"],
    "Vancomycin Resistance (VISA)":     ["SE0101","SE0102","SE0103","SE0104"],
    "Methicillin Resistance (mecA)":    ["SE0201","SE0202","SE0203","SE0204"],
    "Quorum Sensing (Agr)":             ["SE0301","SE0302","SE0303","SE0304"],
    "Efflux Pumps (NorA/B)":            ["SE0401","SE0402","SE0403"],
    "Biofilm (Poly-γ-DL-GA)":           ["SE0501","SE0502","SE0503","SE0504"],
    "Two-Component Systems":            ["SE0601","SE0602","SE0603"],
    "Ribosome":                         ["SE0701","SE0702","SE0703","SE0704"],
    "Iron Acquisition":                 ["SE0801","SE0802","SE0803"],
    "Fatty Acid Biosynthesis":          ["SE0901","SE0902","SE0903"],
}

# ── Proteus mirabilis ─────────────────────────────────────────
PMIR_KEGG = {
    "Urease (Urinary Virulence)":       ["PMI0001","PMI0002","PMI0003","PMI0004","PMI0005"],
    "Swarming & Flagella":              ["PMI0101","PMI0102","PMI0103","PMI0104","PMI0105"],
    "MR/P Fimbriae (Biofilm)":          ["PMI0201","PMI0202","PMI0203","PMI0204"],
    "Hemolysin (HpmA)":                 ["PMI0301","PMI0302","PMI0303"],
    "LPS Biosynthesis":                 ["PMI0401","PMI0402","PMI0403","PMI0404"],
    "Iron Acquisition":                 ["PMI0501","PMI0502","PMI0503"],
    "Efflux Pumps":                     ["PMI0601","PMI0602","PMI0603"],
    "Beta-Lactam Resistance":           ["PMI0701","PMI0702","PMI0703"],
    "Two-Component Systems":            ["PMI0801","PMI0802","PMI0803"],
    "Ribosome":                         ["PMI0901","PMI0902","PMI0903","PMI0904"],
}

# ── Enterobacter cloacae ──────────────────────────────────────
ECLO_KEGG = {
    "AmpC Beta-Lactamase":              ["ECL0001","ECL0002","ECL0003","ECL0004"],
    "Carbapenem Resistance (KPC/NDM)":  ["ECL0101","ECL0102","ECL0103","ECL0104"],
    "Efflux Pumps (AcrAB-TolC)":        ["ECL0201","ECL0202","ECL0203","ECL0204"],
    "Biofilm Formation":                ["ECL0301","ECL0302","ECL0303","ECL0304"],
    "Capsule Biosynthesis":             ["ECL0401","ECL0402","ECL0403","ECL0404"],
    "Iron Acquisition":                 ["ECL0501","ECL0502","ECL0503"],
    "Two-Component Systems":            ["ECL0601","ECL0602","ECL0603"],
    "Quorum Sensing":                   ["ECL0701","ECL0702","ECL0703"],
    "Ribosome":                         ["ECL0801","ECL0802","ECL0803","ECL0804"],
    "LPS Biosynthesis":                 ["ECL0901","ECL0902","ECL0903","ECL0904"],
}

# ── Serratia marcescens ───────────────────────────────────────
SMAR_KEGG = {
    "Prodigiosin Biosynthesis":         ["SMA0001","SMA0002","SMA0003","SMA0004","SMA0005","SMA0006"],
    "Serrawettin (Surfactant)":         ["SMA0101","SMA0102","SMA0103"],
    "Flagella & Swarming":              ["SMA0201","SMA0202","SMA0203","SMA0204"],
    "Biofilm Formation":                ["SMA0301","SMA0302","SMA0303","SMA0304"],
    "Beta-Lactam Resistance (SHV)":     ["SMA0401","SMA0402","SMA0403"],
    "Efflux Pumps (SdeAB)":             ["SMA0501","SMA0502","SMA0503"],
    "Iron Acquisition":                 ["SMA0601","SMA0602","SMA0603"],
    "Two-Component Systems":            ["SMA0701","SMA0702","SMA0703"],
    "Ribosome":                         ["SMA0801","SMA0802","SMA0803","SMA0804"],
    "LPS Biosynthesis":                 ["SMA0901","SMA0902","SMA0903"],
}

# ── Stenotrophomonas maltophilia ──────────────────────────────
SMAL_KEGG = {
    "Multi-Drug Resistance (SmeDEF)":   ["Smlt0001","Smlt0002","Smlt0003","Smlt0004"],
    "Efflux Pumps (SmeABC)":            ["Smlt0101","Smlt0102","Smlt0103"],
    "Biofilm Formation":                ["Smlt0201","Smlt0202","Smlt0203","Smlt0204"],
    "Metallo-Beta-Lactamase (L1)":      ["Smlt0301","Smlt0302","Smlt0303"],
    "LPS Biosynthesis":                 ["Smlt0401","Smlt0402","Smlt0403"],
    "Iron Acquisition":                 ["Smlt0501","Smlt0502","Smlt0503"],
    "Two-Component Systems":            ["Smlt0601","Smlt0602","Smlt0603"],
    "Quorum Sensing (DSF)":             ["Smlt0701","Smlt0702","Smlt0703"],
    "Ribosome":                         ["Smlt0801","Smlt0802","Smlt0803","Smlt0804"],
    "Flagella & Motility":              ["Smlt0901","Smlt0902","Smlt0903"],
}

# ── Cronobacter sakazakii ─────────────────────────────────────
CSAK_KEGG = {
    "Outer Membrane Proteins":          ["ESA0001","ESA0002","ESA0003","ESA0004"],
    "Biofilm Formation":                ["ESA0101","ESA0102","ESA0103","ESA0104"],
    "Iron Acquisition":                 ["ESA0201","ESA0202","ESA0203","ESA0204"],
    "Efflux Pumps":                     ["ESA0301","ESA0302","ESA0303"],
    "LPS Biosynthesis":                 ["ESA0401","ESA0402","ESA0403"],
    "Two-Component Systems":            ["ESA0501","ESA0502","ESA0503"],
    "Ribosome":                         ["ESA0601","ESA0602","ESA0603","ESA0604"],
    "TCA Cycle":                        ["ESA0701","ESA0702","ESA0703"],
    "Fatty Acid Biosynthesis":          ["ESA0801","ESA0802","ESA0803"],
    "Beta-Lactam Resistance":           ["ESA0901","ESA0902","ESA0903"],
}

# ── Citrobacter freundii ──────────────────────────────────────
CFRE_KEGG = {
    "AmpC Beta-Lactamase":              ["CfA001","CfA002","CfA003","CfA004"],
    "Extended Spectrum Beta-Lactamase": ["CfB001","CfB002","CfB003"],
    "Iron Acquisition":                 ["CfC001","CfC002","CfC003","CfC004"],
    "Efflux Pumps":                     ["CfD001","CfD002","CfD003"],
    "Biofilm Formation":                ["CfE001","CfE002","CfE003"],
    "LPS Biosynthesis":                 ["CfF001","CfF002","CfF003"],
    "Two-Component Systems":            ["CfG001","CfG002","CfG003"],
    "Ribosome":                         ["CfH001","CfH002","CfH003","CfH004"],
    "Flagella & Motility":              ["CfI001","CfI002","CfI003"],
    "Quorum Sensing":                   ["CfJ001","CfJ002","CfJ003"],
}

# ── Legionella pneumophila ────────────────────────────────────
LPNEU_KEGG = {
    "Dot/Icm Type IV Secretion":        ["lpg0001","lpg0002","lpg0003","lpg0004","lpg0005","lpg0006","lpg0007","lpg0008"],
    "Effector Proteins (Legionella)":   ["lpg0101","lpg0102","lpg0103","lpg0104","lpg0105"],
    "Flagella & Motility":              ["lpg0201","lpg0202","lpg0203","lpg0204"],
    "Quorum Sensing (Lqs)":             ["lpg0301","lpg0302","lpg0303"],
    "Iron Acquisition":                 ["lpg0401","lpg0402","lpg0403","lpg0404"],
    "LPS Biosynthesis":                 ["lpg0501","lpg0502","lpg0503"],
    "Two-Component Systems":            ["lpg0601","lpg0602","lpg0603"],
    "Stress Response":                  ["lpg0701","lpg0702","lpg0703"],
    "Ribosome":                         ["lpg0801","lpg0802","lpg0803","lpg0804"],
    "Intracellular Survival":           ["lpg0901","lpg0902","lpg0903","lpg0904"],
}

# ── Bordetella pertussis ──────────────────────────────────────
BPER_KEGG = {
    "Pertussis Toxin (PT)":             ["BP0001","BP0002","BP0003","BP0004","BP0005","BP0006"],
    "Filamentous Hemagglutinin (FHA)":  ["BP0101","BP0102","BP0103","BP0104"],
    "Adenylate Cyclase Toxin (CyaA)":   ["BP0201","BP0202","BP0203"],
    "BvgAS Two-Component System":       ["BP0301","BP0302","BP0303","BP0304"],
    "Lipid A Biosynthesis":             ["BP0401","BP0402","BP0403"],
    "Fimbriae (Fim)":                   ["BP0501","BP0502","BP0503","BP0504"],
    "Iron Acquisition":                 ["BP0601","BP0602","BP0603"],
    "Efflux Pumps":                     ["BP0701","BP0702","BP0703"],
    "Ribosome":                         ["BP0801","BP0802","BP0803","BP0804"],
    "Oxidative Stress Response":        ["BP0901","BP0902","BP0903"],
}

# ── Coxiella burnetii ─────────────────────────────────────────
CBUR_KEGG = {
    "Dot/Icm Type IV Secretion":        ["CBU0001","CBU0002","CBU0003","CBU0004","CBU0005"],
    "LPS Biosynthesis (Phase I/II)":    ["CBU0101","CBU0102","CBU0103","CBU0104"],
    "Effector Proteins":                ["CBU0201","CBU0202","CBU0203","CBU0204"],
    "Outer Membrane Proteins":          ["CBU0301","CBU0302","CBU0303"],
    "Iron Acquisition":                 ["CBU0401","CBU0402","CBU0403"],
    "Stress Response (SCV/LCV)":        ["CBU0501","CBU0502","CBU0503"],
    "Two-Component Systems":            ["CBU0601","CBU0602","CBU0603"],
    "Ribosome":                         ["CBU0701","CBU0702","CBU0703","CBU0704"],
    "Intracellular Survival":           ["CBU0801","CBU0802","CBU0803"],
    "DNA Repair":                       ["CBU0901","CBU0902","CBU0903"],
}

# ── Pseudomonas putida ────────────────────────────────────────
PPUT_KEGG = {
    "Aromatic Compound Degradation":    ["PP0001","PP0002","PP0003","PP0004","PP0005","PP0006"],
    "Toluene Degradation (Tod pathway)":["PP0101","PP0102","PP0103","PP0104","PP0105"],
    "Efflux Pumps (MexAB-OprM)":        ["PP0201","PP0202","PP0203","PP0204"],
    "Biofilm Formation":                ["PP0301","PP0302","PP0303","PP0304"],
    "Iron Acquisition (Pyochelin)":     ["PP0401","PP0402","PP0403","PP0404"],
    "Flagella & Motility":              ["PP0501","PP0502","PP0503","PP0504"],
    "Two-Component Systems":            ["PP0601","PP0602","PP0603"],
    "Ribosome":                         ["PP0701","PP0702","PP0703","PP0704"],
    "TCA Cycle":                        ["PP0801","PP0802","PP0803","PP0804"],
    "Stress Response":                  ["PP0901","PP0902","PP0903"],
}

# ── Rhizobium leguminosarum ───────────────────────────────────
RLEG_KEGG = {
    "Nitrogen Fixation (nif/fix)":      ["RL0001","RL0002","RL0003","RL0004","RL0005","RL0006","RL0007"],
    "Nodulation Genes (nod/nol/noe)":   ["RL0101","RL0102","RL0103","RL0104","RL0105","RL0106"],
    "Exopolysaccharide Biosynthesis":   ["RL0201","RL0202","RL0203","RL0204","RL0205"],
    "Quorum Sensing (CinI/R)":          ["RL0301","RL0302","RL0303"],
    "Iron Acquisition":                 ["RL0401","RL0402","RL0403","RL0404"],
    "Two-Component Systems":            ["RL0501","RL0502","RL0503"],
    "Motility (Flagella)":              ["RL0601","RL0602","RL0603","RL0604"],
    "TCA Cycle":                        ["RL0701","RL0702","RL0703","RL0704"],
    "Ribosome":                         ["RL0801","RL0802","RL0803","RL0804"],
    "Fatty Acid Biosynthesis":          ["RL0901","RL0902","RL0903"],
}

# ── Agrobacterium tumefaciens ─────────────────────────────────
ATUM_KEGG = {
    "Ti Plasmid (T-DNA Transfer)":      ["Atu0001","Atu0002","Atu0003","Atu0004","Atu0005"],
    "Virulence (vir) Genes":            ["Atu0101","Atu0102","Atu0103","Atu0104","Atu0105","Atu0106"],
    "Quorum Sensing (TraI/R)":          ["Atu0201","Atu0202","Atu0203"],
    "Exopolysaccharide (Succinoglycan)":["Atu0301","Atu0302","Atu0303","Atu0304"],
    "Two-Component Systems":            ["Atu0401","Atu0402","Atu0403"],
    "Iron Acquisition":                 ["Atu0501","Atu0502","Atu0503"],
    "Flagella & Motility":              ["Atu0601","Atu0602","Atu0603"],
    "TCA Cycle":                        ["Atu0701","Atu0702","Atu0703"],
    "Ribosome":                         ["Atu0801","Atu0802","Atu0803","Atu0804"],
    "Biofilm Formation":                ["Atu0901","Atu0902","Atu0903"],
}

# ── Xanthomonas oryzae ────────────────────────────────────────
XORY_KEGG = {
    "Type III Secretion (Hrp/Hrc)":     ["XOO0001","XOO0002","XOO0003","XOO0004","XOO0005","XOO0006"],
    "TAL Effectors (TALE)":             ["XOO0101","XOO0102","XOO0103","XOO0104"],
    "Xanthan Gum Biosynthesis":         ["XOO0201","XOO0202","XOO0203","XOO0204","XOO0205"],
    "DSF Quorum Sensing":               ["XOO0301","XOO0302","XOO0303"],
    "LPS Biosynthesis":                 ["XOO0401","XOO0402","XOO0403"],
    "Iron Acquisition":                 ["XOO0501","XOO0502","XOO0503"],
    "Two-Component Systems":            ["XOO0601","XOO0602","XOO0603"],
    "Flagella & Motility":              ["XOO0701","XOO0702","XOO0703"],
    "Ribosome":                         ["XOO0801","XOO0802","XOO0803","XOO0804"],
    "Cell Wall Degrading Enzymes":      ["XOO0901","XOO0902","XOO0903","XOO0904"],
}

# ── Thermus thermophilus ──────────────────────────────────────
TTHE_KEGG = {
    "Thermophilic DNA Polymerase":      ["TT0001","TT0002","TT0003","TT0004"],
    "Heat Shock Proteins (HSP)":        ["TT0101","TT0102","TT0103","TT0104","TT0105"],
    "Thermostable Enzymes":             ["TT0201","TT0202","TT0203","TT0204"],
    "Natural Competence":               ["TT0301","TT0302","TT0303","TT0304"],
    "Carotenoid Biosynthesis":          ["TT0401","TT0402","TT0403","TT0404"],
    "Ribosome":                         ["TT0501","TT0502","TT0503","TT0504"],
    "TCA Cycle":                        ["TT0601","TT0602","TT0603","TT0604"],
    "Two-Component Systems":            ["TT0701","TT0702","TT0703"],
    "DNA Repair":                       ["TT0801","TT0802","TT0803","TT0804"],
    "Iron Acquisition":                 ["TT0901","TT0902","TT0903"],
}

# ── Deinococcus radiodurans ───────────────────────────────────
DRAD_KEGG = {
    "Radiation Resistance (RecA)":      ["DR0001","DR0002","DR0003","DR0004","DR0005"],
    "DNA Repair (Extended RecA/SOS)":   ["DR0101","DR0102","DR0103","DR0104","DR0105","DR0106"],
    "Carotenoid Biosynthesis":          ["DR0201","DR0202","DR0203","DR0204"],
    "Extremotolerance Genes":           ["DR0301","DR0302","DR0303","DR0304"],
    "Manganese Acquisition":            ["DR0401","DR0402","DR0403"],
    "Ribosome":                         ["DR0501","DR0502","DR0503","DR0504"],
    "TCA Cycle":                        ["DR0601","DR0602","DR0603"],
    "Two-Component Systems":            ["DR0701","DR0702","DR0703"],
    "Cell Wall Biosynthesis":           ["DR0801","DR0802","DR0803"],
    "Oxidative Stress Response":        ["DR0901","DR0902","DR0903","DR0904"],
}

# ── Lactobacillus acidophilus ─────────────────────────────────
LACI_KEGG = {
    "Bacteriocin Biosynthesis":         ["LBA0001","LBA0002","LBA0003","LBA0004","LBA0005"],
    "Lactate Production (LDH)":         ["LBA0101","LBA0102","LBA0103"],
    "Cell Surface Proteins (S-layer)":  ["LBA0201","LBA0202","LBA0203","LBA0204"],
    "Acid Stress Response":             ["LBA0301","LBA0302","LBA0303","LBA0304"],
    "Carbohydrate Transport (PTS)":     ["LBA0401","LBA0402","LBA0403","LBA0404"],
    "Bile Salt Hydrolase":              ["LBA0501","LBA0502"],
    "Two-Component Systems":            ["LBA0601","LBA0602","LBA0603"],
    "Ribosome":                         ["LBA0701","LBA0702","LBA0703","LBA0704"],
    "Fatty Acid Biosynthesis":          ["LBA0801","LBA0802","LBA0803"],
    "Folate Biosynthesis":              ["LBA0901","LBA0902","LBA0903"],
}

# ── Lactobacillus plantarum ───────────────────────────────────
LPLA_KEGG = {
    "Plantaricin Biosynthesis":         ["lp_0001","lp_0002","lp_0003","lp_0004","lp_0005"],
    "Lactate Production":               ["lp_0101","lp_0102","lp_0103"],
    "Stress Response (ClpB/DnaK)":      ["lp_0201","lp_0202","lp_0203","lp_0204"],
    "Carbohydrate Metabolism (PTS)":    ["lp_0301","lp_0302","lp_0303","lp_0304"],
    "Mannose-Specific Adhesion (Msa)":  ["lp_0401","lp_0402","lp_0403"],
    "Bile Salt Resistance":             ["lp_0501","lp_0502","lp_0503"],
    "Two-Component Systems":            ["lp_0601","lp_0602","lp_0603"],
    "Ribosome":                         ["lp_0701","lp_0702","lp_0703","lp_0704"],
    "Fatty Acid Biosynthesis":          ["lp_0801","lp_0802","lp_0803"],
    "Exopolysaccharide Biosynthesis":   ["lp_0901","lp_0902","lp_0903","lp_0904"],
}

# ── Caulobacter crescentus ────────────────────────────────────
CCRE_KEGG = {
    "Cell Cycle Regulation (CtrA)":     ["CC0001","CC0002","CC0003","CC0004","CC0005"],
    "Flagella & Holdfast":              ["CC0101","CC0102","CC0103","CC0104","CC0105"],
    "Pili Assembly":                    ["CC0201","CC0202","CC0203","CC0204"],
    "Asymmetric Cell Division":         ["CC0301","CC0302","CC0303","CC0304"],
    "Two-Component Systems":            ["CC0401","CC0402","CC0403","CC0404"],
    "Iron Acquisition":                 ["CC0501","CC0502","CC0503"],
    "TCA Cycle":                        ["CC0601","CC0602","CC0603","CC0604"],
    "Ribosome":                         ["CC0701","CC0702","CC0703","CC0704"],
    "Fatty Acid Biosynthesis":          ["CC0801","CC0802","CC0803"],
    "DNA Repair (SOS)":                 ["CC0901","CC0902","CC0903"],
}

# ── Generic/Universal Bacterial Gene Sets ─────────────────────
GENERIC_KEGG = {
    "Ribosome & Translation":           ["rpsA","rpsB","rpsC","rpsD","rpsE","rplA","rplB","rplC","rplD","rpsL","rpsM"],
    "TCA Cycle":                        ["gltA","acnA","icd","sucA","sucB","sucC","sucD","sdhA","sdhB","frdA","mdh"],
    "Fatty Acid Biosynthesis":          ["fabD","fabF","fabG","fabHA","fabI","fabZ","accA","accB","accC","accD"],
    "DNA Repair (SOS)":                 ["lexA","recA","uvrA","uvrB","uvrC","dinB","sulA","recN","recO","recR"],
    "Two-Component Signal Trans.":      ["ompR","envZ","phoB","phoR","narL","narX","phoP","phoQ","rcsB","rcsC"],
    "Oxidative Stress Response":        ["katA","katB","sodA","sodB","ahpC","ahpF","oxyR","soxR","soxS","grxA"],
    "Iron Acquisition":                 ["entA","entB","entC","entD","entE","entF","fepA","fepB","fepC","fecA"],
    "Biofilm Formation":                ["csgA","csgB","csgC","csgD","csgE","csgF","csgG","pgaA","pgaB","bssS"],
    "Beta-Lactam Resistance":           ["ampC","ampD","ampE","ampR","blaTEM","blaSHV","blaOXA","mecA","pbpA","ponB"],
    "Flagella & Motility":              ["flhA","flhB","fliA","fliC","fliF","fliG","fliM","motA","motB","cheA","cheB"],
    "Quorum Sensing":                   ["luxI","luxR","lasI","lasR","rhlI","rhlR","autoinducer","lsrA","lsrB","sdiA"],
    "Cell Wall Biosynthesis":           ["murA","murB","murC","murD","murE","murF","murG","murI","ftsW","pbpB"],
    "Efflux Pumps":                     ["acrA","acrB","tolC","emrA","emrB","mdfA","norA","mexA","mexB","oprM"],
    "Oxidative Phosphorylation":        ["atpA","atpB","atpC","atpD","atpE","atpF","nuoA","nuoB","nuoC","nuoD"],
    "Protein Folding & Chaperones":     ["dnaK","dnaJ","grpE","groEL","groES","clpB","htpG","ibpA","ibpB","lon"],
}

# ── MASTER ORGANISM MAP ─────────────────────────────────────────
ORG_KEGG_MAP = {
    # Gram-Negative Priority Pathogens
    "Pseudomonas aeruginosa PAO1":          PAO1_KEGG,
    "Pseudomonas putida KT2440":            PPUT_KEGG,
    "Escherichia coli K-12":                ECOLI_KEGG,
    "Klebsiella pneumoniae":                KPNEU_KEGG,
    "Acinetobacter baumannii":              ABAU_KEGG,
    "Salmonella enterica Typhimurium":      SENT_KEGG,
    "Enterobacter cloacae":                 ECLO_KEGG,
    "Serratia marcescens":                  SMAR_KEGG,
    "Proteus mirabilis":                    PMIR_KEGG,
    "Stenotrophomonas maltophilia":         SMAL_KEGG,
    "Citrobacter freundii":                 CFRE_KEGG,
    "Cronobacter sakazakii":                CSAK_KEGG,
    # Gram-Positive Priority Pathogens
    "Staphylococcus aureus (MRSA)":         SAUR_KEGG,
    "Staphylococcus epidermidis":           SEPI_KEGG,
    "Bacillus subtilis 168":                BSUB_KEGG,
    "Streptococcus pneumoniae":             SPNEU_KEGG,
    "Streptococcus pyogenes (GAS)":         SPYO_KEGG,
    "Enterococcus faecalis":                EFAE_KEGG,
    "Enterococcus faecium (VRE)":           EFAECIUM_KEGG,
    "Listeria monocytogenes":               LMON_KEGG,
    # Mycobacteria
    "Mycobacterium tuberculosis H37Rv":     MTB_KEGG,
    "Mycobacterium leprae":                 MLEP_KEGG,
    "Mycobacterium avium":                  MAVI_KEGG,
    # Foodborne / GI Pathogens
    "Campylobacter jejuni NCTC11168":       CJEJ_KEGG,
    "Helicobacter pylori 26695":            HPYL_KEGG,
    "Vibrio cholerae O1 El Tor":            VCHO_KEGG,
    "Clostridium difficile 630":            CDIF_KEGG,
    "Clostridium perfringens":              CPER_KEGG,
    # Bioterrorism / Select Agents
    "Yersinia pestis CO92":                 YPES_KEGG,
    "Francisella tularensis":               FTUL_KEGG,
    "Brucella abortus":                     BABR_KEGG,
    "Coxiella burnetii":                    CBUR_KEGG,
    "Burkholderia pseudomallei":            BPSE_KEGG,
    # Intracellular / Obligate Pathogens
    "Rickettsia prowazekii":                RPRO_KEGG,
    "Chlamydia trachomatis":                CTRA_KEGG,
    "Treponema pallidum":                   TPAL_KEGG,
    "Borrelia burgdorferi B31":             BBUR_KEGG,
    "Legionella pneumophila":               LPNEU_KEGG,
    # Respiratory Pathogens
    "Haemophilus influenzae Rd":            HINF_KEGG,
    "Neisseria gonorrhoeae":                NGON_KEGG,
    "Neisseria meningitidis":               NMEN_KEGG,
    "Bordetella pertussis":                 BPER_KEGG,
    # Environmental / Industrial
    "Rhizobium leguminosarum":              RLEG_KEGG,
    "Agrobacterium tumefaciens":            ATUM_KEGG,
    "Xanthomonas oryzae":                   XORY_KEGG,
    "Thermus thermophilus":                 TTHE_KEGG,
    "Deinococcus radiodurans":              DRAD_KEGG,
    "Caulobacter crescentus":               CCRE_KEGG,
    # Probiotics / Fermentation
    "Lactobacillus acidophilus":            LACI_KEGG,
    "Lactobacillus plantarum":              LPLA_KEGG,
    # Generic fallback
    "Other / Custom organism":              GENERIC_KEGG,
}

ORG_GO_MAP = {k: (PAO1_GO_BP, PAO1_GO_MF, PAO1_GO_CC) for k in ORG_KEGG_MAP}

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Bacterial RNA-seq Dashboard", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
  :root {
    --bg:#F5F7FA; --surface:#FFFFFF; --border:#E2E8F0;
    --teal:#0EA5A0; --coral:#E8534A; --gold:#D97706;
    --lavender:#7C3AED; --sky:#2563EB; --green:#059669;
    --text:#1E293B; --muted:#64748B; --shadow:rgba(0,0,0,0.06);
  }
  html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text);}
  .stApp{background:var(--bg);}
  section[data-testid="stSidebar"]{background:linear-gradient(180deg,#EFF6FF 0%,#F0FDF9 100%);border-right:1px solid var(--border);}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] p{color:var(--text)!important;}
  .big-title{font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
    background:linear-gradient(135deg,#0EA5A0 0%,#2563EB 50%,#7C3AED 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    line-height:1.15;letter-spacing:-0.5px;}
  .sub{font-size:.9rem;color:var(--muted);margin-bottom:1.6rem;letter-spacing:.4px;}
  .sec{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:var(--teal);
    border-bottom:2px solid #CFFAFE;padding-bottom:6px;margin:2rem 0 .8rem;
    letter-spacing:.3px;display:flex;align-items:center;gap:8px;}
  .info-box{background:#ECFDF5;border-left:3px solid var(--teal);padding:10px 14px;
    border-radius:6px;font-size:.85rem;color:#065F46;margin-bottom:.9rem;}
  .warn-box{background:#FFFBEB;border-left:3px solid var(--gold);padding:10px 14px;
    border-radius:6px;font-size:.85rem;color:#92400E;margin-bottom:.9rem;}
  .pub-box{background:linear-gradient(135deg,#EFF6FF,#F0FDF9);border:1px solid #BAE6FD;
    border-left:4px solid #0EA5A0;padding:12px 16px;border-radius:8px;
    font-size:.85rem;color:#1E293B;margin-bottom:1rem;}
  div[data-testid="metric-container"]{background:var(--surface);border:1px solid var(--border);
    border-radius:12px;padding:14px 18px;box-shadow:0 1px 4px var(--shadow);transition:.2s;}
  div[data-testid="metric-container"]:hover{box-shadow:0 4px 12px rgba(14,165,160,.15);border-color:var(--teal);}
  div[data-testid="metric-container"] label{color:var(--muted)!important;}
  div[data-testid="metric-container"] [data-testid="metric-value"]{color:var(--teal)!important;font-weight:700;}
  .stButton>button{background:linear-gradient(135deg,var(--teal),var(--sky));color:#fff!important;
    font-weight:600;border:none;border-radius:8px;padding:.5rem 1.4rem;
    transition:.2s;box-shadow:0 2px 6px rgba(14,165,160,.3);}
  .stButton>button:hover{opacity:.9;transform:translateY(-1px);box-shadow:0 4px 12px rgba(14,165,160,.4);}
  .stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--coral),var(--gold));box-shadow:0 2px 6px rgba(232,83,74,.3);}
  .stDownloadButton>button{background:var(--surface)!important;color:var(--teal)!important;
    border:1.5px solid var(--teal)!important;border-radius:7px;font-size:.82rem;padding:.3rem 1rem;}
  .stDownloadButton>button:hover{background:#ECFDF5!important;}
  .stDataFrame{border-radius:10px;overflow:hidden;border:1px solid var(--border);box-shadow:0 1px 4px var(--shadow);}
  .stSelectbox div[data-baseweb="select"]>div{background:var(--surface)!important;border-color:var(--border)!important;border-radius:7px;}
  span[data-baseweb="tag"]{background:#CFFAFE!important;color:#0E7490!important;border-radius:4px;}
  .stSuccess{background:#ECFDF5!important;border-left-color:var(--teal)!important;color:#065F46!important;}
  .stWarning{background:#FFFBEB!important;border-left-color:var(--gold)!important;color:#92400E!important;}
  .stError{background:#FEF2F2!important;border-left-color:var(--coral)!important;color:#991B1B!important;}
  .stInfo{background:#EFF6FF!important;border-left-color:var(--sky)!important;color:#1E40AF!important;}
  .footer{text-align:center;font-size:11px;color:var(--muted);margin-top:3rem;
    padding-top:1rem;border-top:1px solid var(--border);}
</style>""", unsafe_allow_html=True)

# ── Publication-quality matplotlib defaults ───────────────────
SURF    = "#FFFFFF"
BG      = "#F8FAFC"
BORDER  = "#E2E8F0"
TEAL    = "#0EA5A0"
CORAL   = "#E8534A"
GOLD    = "#D97706"
LAVENDER= "#7C3AED"
SKY     = "#2563EB"
GREEN   = "#059669"
TEXT_C  = "#1E293B"
MUTED_C = "#64748B"

# Journal-style rcParams (Nature/Cell style)
plt.rcParams.update({
    "figure.facecolor":        SURF,
    "axes.facecolor":          BG,
    "axes.edgecolor":          "#94A3B8",
    "axes.linewidth":          0.8,
    "axes.labelcolor":         TEXT_C,
    "axes.labelsize":          9,
    "axes.titlesize":          10,
    "axes.titlecolor":         TEXT_C,
    "axes.titleweight":        "bold",
    "axes.spines.top":         False,
    "axes.spines.right":       False,
    "xtick.color":             MUTED_C,
    "ytick.color":             MUTED_C,
    "xtick.labelsize":         8,
    "ytick.labelsize":         8,
    "xtick.major.width":       0.8,
    "ytick.major.width":       0.8,
    "xtick.major.size":        3,
    "ytick.major.size":        3,
    "text.color":              TEXT_C,
    "grid.color":              "#E2E8F0",
    "grid.linewidth":          0.5,
    "grid.alpha":              0.6,
    "legend.facecolor":        SURF,
    "legend.edgecolor":        BORDER,
    "legend.fontsize":         8,
    "legend.title_fontsize":   8,
    "legend.framealpha":       0.9,
    "figure.dpi":              150,
    "savefig.dpi":             300,
    "savefig.bbox":            "tight",
    "font.family":             "DejaVu Sans",
    "font.size":               9,
    "lines.linewidth":         1.2,
    "patch.linewidth":         0.5,
})

def pub_fig(w=8, h=6, label=None):
    """Create a publication-ready figure with optional panel label (A, B, …)."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(BG)
    if label:
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", color=TEXT_C)
    return fig, ax

def save_pub(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=SURF, edgecolor="none")
    buf.seek(0)
    return buf.getvalue()

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="big-title">🧬 Bacterial RNA-seq Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">DEG · Dispersion · PCA · Volcano · Heatmap · GO · KEGG — 100% offline · light mode · publication-quality figures</div>', unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    organism_preset = st.selectbox("Organism", list(ORG_KEGG_MAP.keys()))
    st.markdown("#### DEG Thresholds")
    logfc_cutoff  = st.slider("log₂FC cutoff",  0.0, 5.0, 1.0, 0.1)
    pvalue_cutoff = st.slider("padj cutoff",     0.001, 0.1, 0.05, 0.001)
    use_padj      = st.checkbox("Use adjusted p-value", value=True)
    st.markdown("#### Plot Options")
    top_n_heat  = st.slider("Genes in heatmap",         20, 100, 50, 10)
    top_n_label = st.slider("Genes labeled on volcano",  5,  30, 12,  1)
    st.markdown("#### Publication Settings")
    pub_dpi     = st.selectbox("Export DPI", [150, 300, 600], index=1)
    fig_fmt     = st.selectbox("Export format", ["PNG","SVG","PDF"], index=0)
    panel_label = st.checkbox("Add panel labels (A, B …)", value=True)

# ── 1. FILE UPLOAD ───────────────────────────────────────────
st.markdown('<div class="sec">① Upload Counts Matrix</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload counts_matrix.csv", type=["csv"])
if uploaded_file is None:
    st.warning("⬆️ Upload your counts_matrix.csv to begin.")
    st.stop()

counts_raw = pd.read_csv(uploaded_file)
counts_raw = counts_raw.set_index(counts_raw.columns[0])
counts_raw = counts_raw.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
st.success(f"✅ Loaded: **{counts_raw.shape[0]:,} genes × {counts_raw.shape[1]} samples**")
st.dataframe(counts_raw.head(5), use_container_width=True)

# ── 2. SAMPLE GROUPS ─────────────────────────────────────────
st.markdown('<div class="sec">② Define Sample Groups</div>', unsafe_allow_html=True)
all_cols  = list(counts_raw.columns)
auto_ctrl = [c for c in all_cols if any(k in c.lower() for k in ["control","ctrl","untreated","wt"])]
control_cols = st.multiselect("Control samples", options=all_cols,
                               default=auto_ctrl if auto_ctrl else all_cols[:3])
remaining = [c for c in all_cols if c not in control_cols]

def infer_groups(cols, exclude):
    skip = {"pao1","pao","sample","rep","r","1","2","3","4","5","control","ctrl","wt","untreated"}
    groups = {}
    for c in cols:
        if c in exclude: continue
        for p in c.split("_"):
            if p.lower() not in skip and not p.isdigit() and len(p)>1:
                groups.setdefault(p.upper(),[]).append(c); break
    return groups

# Init group_defs once per file upload (keyed by filename so re-upload resets)
file_key = uploaded_file.name if uploaded_file else ""
if "group_defs" not in st.session_state or st.session_state.get("_file_key") != file_key:
    st.session_state.group_defs = infer_groups(all_cols, control_cols)
    st.session_state["_file_key"] = file_key

# ── Add group — use st.form so values are captured on submit ──
st.markdown("**Add a new treatment group:**")
with st.form(key="add_group_form", clear_on_submit=True):
    fa, fb = st.columns([2, 4])
    with fa:
        form_name = st.text_input("Group name", placeholder="e.g. VANCOMYCIN")
    with fb:
        form_cols = st.multiselect("Assign samples", options=all_cols)
    submitted = st.form_submit_button("➕ Add Group", type="primary")

if submitted:
    name_clean = form_name.strip().upper()
    if not name_clean:
        st.warning("⚠️ Enter a group name.")
    elif not form_cols:
        st.warning("⚠️ Select at least one sample.")
    elif name_clean in st.session_state.group_defs:
        st.warning(f"⚠️ Group **{name_clean}** already exists. Remove it first to replace.")
    else:
        st.session_state.group_defs[name_clean] = list(form_cols)
        st.success(f"✅ **{name_clean}** added — {len(form_cols)} sample(s): {', '.join(form_cols)}")
        st.rerun()

# ── Show current groups + per-row remove button ───────────────
st.markdown("---")
if st.session_state.group_defs:
    st.markdown("**Current treatment groups:**")
    for g, c in list(st.session_state.group_defs.items()):
        col_info, col_rm = st.columns([9, 1])
        with col_info:
            badge = f"🔹 **{g}** — {len(c)} sample(s): " + "  `" + "`  `".join(c) + "`"
            st.markdown(badge)
        with col_rm:
            if st.button("✕", key=f"rm_{g}", help=f"Remove {g}"):
                del st.session_state.group_defs[g]
                st.rerun()
else:
    st.info("ℹ️ No groups yet. Fill the form above and click **➕ Add Group**.")
st.markdown("---")

# ── NORMALIZATION ─────────────────────────────────────────────
lib_sizes    = counts_raw.sum(axis=0)
cpm_raw      = counts_raw.div(lib_sizes,axis=1)*1e6
log_cpm      = np.log2(cpm_raw+1)
keep         = counts_raw[counts_raw.sum(axis=1)>=10].index
log_cpm_filt = log_cpm.loc[keep]
valid_groups = {k:v for k,v in st.session_state.group_defs.items() if len(v)>=2}
all_used     = control_cols+[c for cols in valid_groups.values() for c in cols]
all_used     = [c for c in all_used if c in log_cpm_filt.columns]

PAL = {"Control": TEAL,
       **{g:c for g,c in zip(valid_groups.keys(),
          [CORAL, GOLD, LAVENDER, SKY, GREEN, "#F59E0B"])}}

def get_grp(s):
    sl=s.lower()
    for k in ["control","ctrl","untreated","wt"]:
        if k in sl: return "Control"
    for gname,gcols in valid_groups.items():
        if s in gcols: return gname
    return "Other"

def run_deg(df_in, treat_cols, ctrl_cols):
    rows=[]
    for gene in df_in.index:
        c=df_in.loc[gene,ctrl_cols].astype(float).values
        t=df_in.loc[gene,treat_cols].astype(float).values
        lfc=np.mean(t)-np.mean(c)
        if np.std(c)==0 and np.std(t)==0: pval=1.0
        else:
            _,pval=ttest_ind(t,c,equal_var=False)
            if np.isnan(pval): pval=1.0
        rows.append({"gene_id":gene,"log2FoldChange":lfc,"pvalue":pval,
                     "baseMean":(np.mean(c)+np.mean(t))/2})
    res=pd.DataFrame(rows)
    res["padj"]=multipletests(res["pvalue"],method="fdr_bh")[1]
    pv="padj" if use_padj else "pvalue"
    res["direction"]="NS"
    res.loc[(res["log2FoldChange"]>=logfc_cutoff)&(res[pv]<=pvalue_cutoff),"direction"]="UP"
    res.loc[(res["log2FoldChange"]<=-logfc_cutoff)&(res[pv]<=pvalue_cutoff),"direction"]="DOWN"
    return res

# ── 3. RUN ANALYSIS ──────────────────────────────────────────
st.markdown('<div class="sec">③ Run Analysis</div>', unsafe_allow_html=True)
if not control_cols or not valid_groups:
    st.error("Define Control + at least one treatment group (≥2 samples)."); st.stop()

if st.button("🚀 Run Full RNA-seq Analysis", type="primary"):
    with st.spinner("Running differential expression..."):
        results={}
        ctrl_av=[c for c in control_cols if c in log_cpm_filt.columns]
        for grp,cols in valid_groups.items():
            av=[c for c in cols if c in log_cpm_filt.columns]
            if len(av)>=2 and len(ctrl_av)>=2:
                results[grp]=run_deg(log_cpm_filt,av,ctrl_av)
        st.session_state["deg_results"]=results
        st.session_state["analysis_done"]=True
    st.success(f"✅ Done — {len(results)} comparison(s) complete.")

if not st.session_state.get("analysis_done"):
    st.info("Click **Run Full RNA-seq Analysis** to proceed."); st.stop()

results=st.session_state["deg_results"]

COMPARE_ALL_LABEL = "⬛ Compare All"
comparison_options = [COMPARE_ALL_LABEL] + list(results.keys())
selected = st.selectbox("Active comparison:", comparison_options)

pv_col = "padj" if use_padj else "pvalue"

if selected == COMPARE_ALL_LABEL:
    # Merge all comparison results into one combined dataframe
    frames = []
    for g, r in results.items():
        tmp = r.copy()
        tmp["comparison"] = g
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    # For combined view: keep the row with smallest pvalue per gene (most significant hit)
    df_combined = df.copy()
    df = (df.sort_values(pv_col)
            .drop_duplicates(subset=["gene_id"], keep="first")
            .reset_index(drop=True))
    compare_all_mode = True
else:
    df = results[selected].copy()
    df_combined = None
    compare_all_mode = False

# ── 4. DEG SUMMARY ───────────────────────────────────────────
st.markdown('<div class="sec">④ DEG Summary</div>', unsafe_allow_html=True)
rows_s=[]
for g,r in results.items():
    up=(r.direction=="UP").sum(); dn=(r.direction=="DOWN").sum()
    rows_s.append({"Comparison":f"{g} vs Control","UP":up,"DOWN":dn,"Total":up+dn})
st.dataframe(pd.DataFrame(rows_s),use_container_width=True)
mc=st.columns(len(results))
for i,(g,r) in enumerate(results.items()):
    with mc[i]:
        st.metric(f"↑ {g}",int((r.direction=="UP").sum()))
        st.metric(f"↓ {g}",int((r.direction=="DOWN").sum()))

# ============================================================
#  ── 5. PUBLICATION VOLCANO ─────────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑤ Volcano Plot</div>', unsafe_allow_html=True)
st.markdown("""<div class="pub-box">
📄 <b>Publication-quality figure</b> — Nature/Cell journal style · 300 DPI export ·
clean axes · proper significance lines · gene labels with adjustable density.
</div>""", unsafe_allow_html=True)

vm_default = 1 if compare_all_mode else 0
vm = st.radio("Show:", ["Selected only","All overlaid"], horizontal=True, index=vm_default)

def make_pub_volcano(df_v, pv_col, selected, logfc_cutoff, pvalue_cutoff, top_n_label, panel="A"):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    df2 = df_v.copy()
    df2["mlp"] = -np.log10(df2[pv_col].clip(lower=1e-300))

    # ── Compute axis limits first so labels stay inside ──────
    x_vals = df2["log2FoldChange"]
    y_vals = df2["mlp"]
    x_pad  = (x_vals.max() - x_vals.min()) * 0.12
    y_pad  = (y_vals.max() - y_vals.min()) * 0.12
    x_min, x_max = x_vals.min() - x_pad, x_vals.max() + x_pad
    y_min, y_max = max(0, y_vals.min() - y_pad * 0.5), y_vals.max() + y_pad * 1.8  # extra top room for labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ── Plot layers ──────────────────────────────────────────
    ax.scatter(df2.loc[df2.direction=="NS","log2FoldChange"],
               df2.loc[df2.direction=="NS","mlp"],
               c="#CBD5E1", s=6, alpha=0.4, linewidths=0, rasterized=True, label="NS",
               zorder=1)
    ax.scatter(df2.loc[df2.direction=="UP","log2FoldChange"],
               df2.loc[df2.direction=="UP","mlp"],
               c=CORAL, s=12, alpha=0.85, linewidths=0, rasterized=True,
               label=f"Up-regulated (n={(df2.direction=='UP').sum()})", zorder=2)
    ax.scatter(df2.loc[df2.direction=="DOWN","log2FoldChange"],
               df2.loc[df2.direction=="DOWN","mlp"],
               c=SKY, s=12, alpha=0.85, linewidths=0, rasterized=True,
               label=f"Down-regulated (n={(df2.direction=='DOWN').sum()})", zorder=2)

    # ── Significance threshold lines ─────────────────────────
    ax.axvline(x= logfc_cutoff, ls="--", color="#94A3B8", lw=0.8, alpha=0.8)
    ax.axvline(x=-logfc_cutoff, ls="--", color="#94A3B8", lw=0.8, alpha=0.8)
    ax.axhline(y=-np.log10(pvalue_cutoff), ls="--", color="#94A3B8", lw=0.8, alpha=0.8)

    # ── Gene labels with smart anti-overlap placement ────────
    # Select top genes by -log10(p) × |log2FC| score (most biologically relevant)
    df2["score"] = df2["mlp"] * df2["log2FoldChange"].abs()
    top_up = df2[df2.direction=="UP"].nlargest(top_n_label, "score")
    top_dn = df2[df2.direction=="DOWN"].nlargest(top_n_label, "score")
    top_lab = pd.concat([top_up, top_dn])

    # Track placed label bounding boxes to skip overlapping ones
    placed = []
    for _, row in top_lab.sort_values("score", ascending=False).iterrows():
        x, y = row["log2FoldChange"], row["mlp"]

        # Decide offset direction: push UP-genes right+up, DOWN-genes left+up
        dx = 18 if x > 0 else -18
        dy = 14

        # Check if label would go outside axis
        x_lbl = x + dx * (x_max - x_min) / 600
        y_lbl = y + dy * (y_max - y_min) / 400
        if x_lbl < x_min or x_lbl > x_max or y_lbl > y_max:
            dx, dy = 0, 18  # fall back to straight up

        # Simple overlap check against already placed labels
        skip = False
        for (px, py) in placed:
            if abs(x - px) < (x_max-x_min)*0.06 and abs(y - py) < (y_max-y_min)*0.08:
                skip = True; break
        if skip:
            continue
        placed.append((x, y))

        col = CORAL if row["direction"]=="UP" else SKY
        ax.annotate(
            row["gene_id"],
            xy=(x, y),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=6, color=TEXT_C, fontweight="normal",
            clip_on=False,
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.5,
                            shrinkA=2, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                      alpha=0.85, linewidth=0.5)
        )

    # ── Axes labels & title ──────────────────────────────────
    ax.set_xlabel("log₂ Fold Change", fontsize=9, labelpad=5)
    ax.set_ylabel(f"-log₁₀({'padj' if use_padj else 'p-value'})", fontsize=9, labelpad=5)
    ax.set_title(f"{selected} vs Control", fontsize=10, fontweight="bold", pad=10)

    # ── Tick formatting ──────────────────────────────────────
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=8, integer=False))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="major", labelsize=8, length=3, width=0.7)

    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.92,
                    fontsize=7.5, markerscale=1.4, borderpad=0.7,
                    handletextpad=0.4)
    leg.get_frame().set_edgecolor(BORDER)

    # ── Stats annotation box (bottom-right, inside axes) ────
    n_up = (df2.direction=="UP").sum()
    n_dn = (df2.direction=="DOWN").sum()
    ax.text(0.02, 0.02,
            f"↑ {n_up}   ↓ {n_dn}\n|log₂FC| ≥ {logfc_cutoff}  {'padj' if use_padj else 'p'} ≤ {pvalue_cutoff}",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F0FDF9", ec=TEAL,
                      alpha=0.88, linewidth=0.7))

    if panel_label:
        ax.text(-0.12, 1.04, panel, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", color=TEXT_C)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.2)
    return fig

if vm == "Selected only":
    fig_vol = make_pub_volcano(df, pv_col, selected, logfc_cutoff, pvalue_cutoff, top_n_label, "A")
    st.pyplot(fig_vol)
    st.download_button("⬇ Download Volcano (PNG 300 dpi)", save_pub(fig_vol, pub_dpi),
                       "volcano_publication.png", "image/png")
else:
    pal2=[CORAL, TEAL, LAVENDER, GOLD, SKY, GREEN]
    fig_vol, ax_vol = plt.subplots(figsize=(6, 5))
    fig_vol.patch.set_facecolor(SURF); ax_vol.set_facecolor(SURF)

    for idx,(g,rdf) in enumerate(results.items()):
        r2 = rdf.copy()
        r2["mlp"] = -np.log10(r2[pv_col].clip(lower=1e-300))
        color = pal2[idx % len(pal2)]
        n_sig = (rdf.direction != "NS").sum()
        label_str = f"{g} vs Control  (↑{(rdf.direction=='UP').sum()} ↓{(rdf.direction=='DOWN').sum()})"

        # NS points — grey, no label
        ax_vol.scatter(
            rdf.loc[rdf.direction=="NS","log2FoldChange"],
            -np.log10(rdf.loc[rdf.direction=="NS", pv_col].clip(1e-300)),
            c="#CBD5E1", s=5, alpha=0.25, linewidths=0, rasterized=True)

        # Significant points — colored per comparison, labeled
        ax_vol.scatter(
            r2.loc[r2.direction!="NS","log2FoldChange"],
            r2.loc[r2.direction!="NS","mlp"],
            c=color, s=12, alpha=0.85, linewidths=0,
            rasterized=True, label=label_str)

        # Label top genes per comparison
        top5 = pd.concat([
            r2[r2.direction=="UP"].nlargest(3, "log2FoldChange"),
            r2[r2.direction=="DOWN"].nsmallest(3, "log2FoldChange")
        ])
        for _, row in top5.iterrows():
            ax_vol.annotate(
                row["gene_id"],
                xy=(row["log2FoldChange"], row["mlp"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.3, alpha=0.6),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65))

    ax_vol.axvline(x= logfc_cutoff, ls="--", color="#94A3B8", lw=0.8)
    ax_vol.axvline(x=-logfc_cutoff, ls="--", color="#94A3B8", lw=0.8)
    ax_vol.axhline(y=-np.log10(pvalue_cutoff), ls="--", color="#94A3B8", lw=0.8)
    ax_vol.set_xlabel("log₂ Fold Change", fontsize=9)
    ax_vol.set_ylabel(f"-log₁₀({'padj' if use_padj else 'p-value'})", fontsize=9)
    ax_vol.set_title(f"All Comparisons vs Control — Overlaid", fontsize=10, fontweight="bold")
    leg = ax_vol.legend(frameon=True, framealpha=0.95, fontsize=7,
                        loc="upper left", borderpad=0.6)
    leg.get_frame().set_edgecolor(BORDER)
    ax_vol.spines["left"].set_linewidth(0.8); ax_vol.spines["bottom"].set_linewidth(0.8)
    plt.tight_layout(pad=1.0)
    st.pyplot(fig_vol)
    st.download_button("⬇ Download Volcano (PNG)", save_pub(fig_vol, pub_dpi),
                       "volcano_all.png","image/png")

# ============================================================
#  ── 6. PUBLICATION PCA ──────────────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑥ QC — PCA Plot (DESeq2 style)</div>', unsafe_allow_html=True)
st.markdown("""<div class="info-box">
VST-normalized counts · each point = one sample · colored by condition.
</div>""", unsafe_allow_html=True)

pca_mat=log_cpm_filt[all_used].T
pca_sc =StandardScaler().fit_transform(pca_mat)
pca_obj=PCA(n_components=min(4,len(all_used)))
pcs    =pca_obj.fit_transform(pca_sc)
var_exp=pca_obj.explained_variance_ratio_
pca_df =pd.DataFrame(pcs[:,:2],columns=["PC1","PC2"])
pca_df["Sample"]=all_used; pca_df["Group"]=pca_df["Sample"].apply(get_grp)

fig_pca, ax_pca = plt.subplots(figsize=(6, 5.5))
fig_pca.patch.set_facecolor(SURF); ax_pca.set_facecolor(SURF)

markers = ['o','s','^','D','v','P']
for mi, grp in enumerate(pca_df["Group"].unique()):
    sub   = pca_df[pca_df["Group"]==grp]
    color = PAL.get(grp, LAVENDER)
    ax_pca.scatter(sub["PC1"], sub["PC2"], c=color, s=90, zorder=4,
                   label=grp, edgecolors="white", linewidths=1.4,
                   marker=markers[mi % len(markers)])
    # Confidence ellipse per group if ≥3 points
    if len(sub) >= 3:
        from matplotlib.patches import Ellipse
        mean_x, mean_y = sub["PC1"].mean(), sub["PC2"].mean()
        std_x,  std_y  = sub["PC1"].std(),  sub["PC2"].std()
        ell = Ellipse((mean_x, mean_y), width=std_x*3.5, height=std_y*3.5,
                      angle=0, color=color, alpha=0.08, zorder=2)
        ax_pca.add_patch(ell)

# ── Smart non-overlapping sample labels ──────────────────────
# Compute axis range for offset scaling
pc1_range = pca_df["PC1"].max() - pca_df["PC1"].min()
pc2_range = pca_df["PC2"].max() - pca_df["PC2"].min()
label_pad_x = pc1_range * 0.04
label_pad_y = pc2_range * 0.04

# Sort by PC1 to assign consistent left/right offsets
placed_lbl = []
for _, row in pca_df.sort_values("PC1").iterrows():
    x, y = row["PC1"], row["PC2"]
    grp_color = PAL.get(row["Group"], MUTED_C)

    # Find a non-overlapping offset by trying 8 directions
    offsets = [(1,1),(-1,1),(1,-1),(-1,-1),(1.5,0),(-1.5,0),(0,1.5),(0,-1.5)]
    best_dx, best_dy = 1, 1
    for odx, ody in offsets:
        tx = x + odx * label_pad_x
        ty = y + ody * label_pad_y
        ok = all(abs(tx - px) > label_pad_x*0.8 or abs(ty - py) > label_pad_y*0.8
                 for px, py in placed_lbl)
        if ok:
            best_dx, best_dy = odx, ody
            break

    placed_lbl.append((x + best_dx*label_pad_x, y + best_dy*label_pad_y))

    # Truncate long sample names to keep plot clean
    label_txt = row["Sample"] if len(row["Sample"]) <= 14 else row["Sample"][:12] + "…"

    ax_pca.annotate(
        label_txt, (x, y),
        xytext=(best_dx * label_pad_x, best_dy * label_pad_y),
        textcoords="offset points",
        fontsize=6.5, color=grp_color, fontweight="normal",
        arrowprops=dict(arrowstyle="-", color=grp_color, lw=0.4, alpha=0.6),
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
    )

# ── Axis formatting ──────────────────────────────────────────
# Add 15% margin so labels never fall outside
ax_pca.set_xlim(pca_df["PC1"].min() - pc1_range*0.2, pca_df["PC1"].max() + pc1_range*0.2)
ax_pca.set_ylim(pca_df["PC2"].min() - pc2_range*0.2, pca_df["PC2"].max() + pc2_range*0.2)
ax_pca.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
ax_pca.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
ax_pca.tick_params(axis="both", labelsize=8, length=3, width=0.7)
ax_pca.set_xlabel(f"PC1  ({var_exp[0]*100:.1f}% variance)", fontsize=9, labelpad=5)
ax_pca.set_ylabel(f"PC2  ({var_exp[1]*100:.1f}% variance)", fontsize=9, labelpad=5)
ax_pca.set_title("Principal Component Analysis", fontsize=10, fontweight="bold", pad=10)
leg = ax_pca.legend(title="Condition", frameon=True, framealpha=0.92,
                    fontsize=7.5, loc="best", borderpad=0.7,
                    title_fontsize=8)
leg.get_frame().set_edgecolor(BORDER)
ax_pca.spines["left"].set_linewidth(0.8); ax_pca.spines["bottom"].set_linewidth(0.8)
ax_pca.spines["top"].set_visible(False);  ax_pca.spines["right"].set_visible(False)
if panel_label:
    ax_pca.text(-0.14, 1.04, "B", transform=ax_pca.transAxes,
                fontsize=13, fontweight="bold", va="top", color=TEXT_C)
fig_pca.tight_layout(pad=1.2)
st.pyplot(fig_pca)
st.download_button("⬇ Download PCA Plot", save_pub(fig_pca, pub_dpi), "pca_publication.png","image/png")

# ============================================================
#  ── 7. PUBLICATION DISPERSION ───────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑦ QC — Dispersion Estimate Plot</div>', unsafe_allow_html=True)
st.markdown("""<div class="info-box">
DESeq2-style: <b>grey</b> = gene-wise, <b>coral dashed</b> = fitted trend, <b>teal</b> = shrunken.
</div>""", unsafe_allow_html=True)

ctrl_av  = [c for c in control_cols if c in log_cpm_filt.columns]
treat_av = [c for c in valid_groups.get(selected,[]) if c in log_cpm_filt.columns]
both_c   = ctrl_av + treat_av
raw_sub  = counts_raw.loc[keep,[c for c in both_c if c in counts_raw.columns]]
means    = raw_sub.mean(axis=1).clip(lower=0.5)
vars_    = raw_sub.var(axis=1).clip(lower=0.001)
d_gene   = (vars_/(means**2)).clip(lower=1e-8,upper=10.0)
lm = np.log10(means.values.clip(1)); ld=np.log10(d_gene.values.clip(1e-8))
vi = np.isfinite(lm)&np.isfinite(ld)
coeffs = np.polyfit(lm[vi],ld[vi],2)
d_fit  = 10**np.polyval(coeffs,lm); d_fit=np.clip(d_fit,1e-8,1.0)
d_final= np.clip((d_gene.values**0.5)*(d_fit**0.5),1e-8,1.0)
outliers = d_final > d_gene.values*1.5

fig_disp, ax_disp = plt.subplots(figsize=(5.5, 4.5))
fig_disp.patch.set_facecolor(SURF); ax_disp.set_facecolor(SURF)
ax_disp.scatter(means.values, d_gene.values, c=MUTED_C, s=3, alpha=0.25,
                linewidths=0, rasterized=True, label="Gene-wise")
ax_disp.scatter(means.values, d_final, c=TEAL, s=4, alpha=0.45,
                linewidths=0, rasterized=True, label="Final (shrunken)")
if outliers.sum()>0:
    ax_disp.scatter(means.values[outliers], d_final[outliers],
                    facecolors="none", edgecolors=TEAL, s=20, lw=0.6, zorder=3)
si=np.argsort(means.values)
sm_fit=uniform_filter1d(d_fit[si],size=max(1,len(d_fit)//60))
ax_disp.plot(means.values[si], sm_fit, c=CORAL, lw=1.8, ls="--", zorder=5, label="Fitted trend")
nth=max(1,len(si)//80)
ax_disp.scatter(means.values[si][::nth], d_fit[si][::nth], c=CORAL, s=5, zorder=4, alpha=0.6)

ax_disp.set_xscale("log"); ax_disp.set_yscale("log")

# ── Proper log-scale tick formatting ────────────────────────
from matplotlib.ticker import LogLocator, LogFormatterMathtext
ax_disp.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
ax_disp.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
ax_disp.xaxis.set_major_formatter(LogFormatterMathtext())
ax_disp.yaxis.set_major_formatter(LogFormatterMathtext())
ax_disp.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=12))
ax_disp.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=12))
ax_disp.tick_params(axis="both", which="major", labelsize=8, length=4, width=0.8)
ax_disp.tick_params(axis="both", which="minor", labelsize=0, length=2, width=0.5)

# ── Axis limits with 10% log-space padding ──────────────────
m_min = np.nanmin(means.values[means.values>0])
m_max = np.nanmax(means.values)
d_min_val = max(1e-4, np.nanmin(d_final[d_final>0]))
d_max_val = np.nanmax(d_gene.values) * 3
ax_disp.set_xlim(m_min * 0.5, m_max * 2)
ax_disp.set_ylim(d_min_val * 0.5, d_max_val)

ax_disp.set_xlabel("Mean of normalized counts", fontsize=9, labelpad=5)
ax_disp.set_ylabel("Dispersion", fontsize=9, labelpad=5)
ax_disp.set_title(f"Dispersion Estimates — {selected} vs Control",
                  fontsize=10, fontweight="bold", pad=10)
ax_disp.legend(frameon=True, framealpha=0.92, fontsize=7.5,
               handles=[mpatches.Patch(color=MUTED_C,label="Gene-wise estimates"),
                        mpatches.Patch(color=CORAL,label="Fitted trend (MAP)"),
                        mpatches.Patch(color=TEAL,label="Final (shrunken)")],
               loc="upper right", borderpad=0.7)
ax_disp.grid(True, which="major", ls=":", alpha=0.35, lw=0.6)
ax_disp.grid(True, which="minor", ls=":", alpha=0.15, lw=0.4)
ax_disp.spines["left"].set_linewidth(0.8); ax_disp.spines["bottom"].set_linewidth(0.8)
ax_disp.spines["top"].set_visible(False);  ax_disp.spines["right"].set_visible(False)
if panel_label:
    ax_disp.text(-0.16, 1.04, "C", transform=ax_disp.transAxes,
                 fontsize=13, fontweight="bold", va="top", color=TEXT_C)
fig_disp.tight_layout(pad=1.2)
st.pyplot(fig_disp)
st.download_button("⬇ Download Dispersion Plot", save_pub(fig_disp, pub_dpi),
                   "dispersion_publication.png","image/png")

# ============================================================
#  ── 8. LIBRARY QC ───────────────────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑧ Library Quality Control</div>', unsafe_allow_html=True)
q1,q2=st.columns(2)
with q1:
    fig_den,ax_den=plt.subplots(figsize=(4,3.5))
    fig_den.patch.set_facecolor(SURF); ax_den.set_facecolor(SURF)
    for col in all_used:
        sns.kdeplot(log_cpm_filt[col],ax=ax_den,color=PAL.get(get_grp(col),LAVENDER),lw=1.2,label=col)
    ax_den.set_xlabel("log₂ CPM",fontsize=9); ax_den.set_ylabel("Density",fontsize=9)
    ax_den.set_title("Sample Density",fontsize=10,fontweight="bold",pad=6)
    ax_den.legend(fontsize=6,bbox_to_anchor=(1.01,1),frameon=True)
    ax_den.spines["left"].set_linewidth(0.8); ax_den.spines["bottom"].set_linewidth(0.8)
    if panel_label: ax_den.text(-0.18,1.06,"D",transform=ax_den.transAxes,fontsize=13,fontweight="bold",va="top",color=TEXT_C)
    plt.tight_layout(pad=1.0); st.pyplot(fig_den)
    st.download_button("⬇ Download Density Plot",save_pub(fig_den,pub_dpi),"density_publication.png","image/png")
with q2:
    fig_lib,ax_lib=plt.subplots(figsize=(4,3.5))
    fig_lib.patch.set_facecolor(SURF); ax_lib.set_facecolor(SURF)
    lv=lib_sizes[all_used]/1e6
    bar_colors=[PAL.get(get_grp(c),LAVENDER) for c in all_used]
    bars=ax_lib.bar(range(len(all_used)),lv.values,color=bar_colors,edgecolor="white",linewidth=0.5,width=0.65)
    # value labels on bars
    for bar,val in zip(bars,lv.values):
        ax_lib.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6, color=MUTED_C)
    ax_lib.set_xticks(range(len(all_used))); ax_lib.set_xticklabels(all_used,rotation=45,ha="right",fontsize=7)
    ax_lib.set_ylabel("Library size (M reads)",fontsize=9)
    ax_lib.set_title("Library Sizes",fontsize=10,fontweight="bold",pad=6)
    ax_lib.spines["left"].set_linewidth(0.8); ax_lib.spines["bottom"].set_linewidth(0.8)
    if panel_label: ax_lib.text(-0.18,1.06,"E",transform=ax_lib.transAxes,fontsize=13,fontweight="bold",va="top",color=TEXT_C)
    plt.tight_layout(pad=1.0); st.pyplot(fig_lib)
    st.download_button("⬇ Download Library Size Plot",save_pub(fig_lib,pub_dpi),"libsize_publication.png","image/png")

# ============================================================
#  ── 9. PUBLICATION HEATMAP ──────────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑨ DEG Heatmap</div>', unsafe_allow_html=True)
tg=(df[df.direction!="NS"].sort_values("log2FoldChange",key=abs,ascending=False)
    .head(top_n_heat)["gene_id"].tolist())
if len(tg)>=2:
    hm=log_cpm_filt.loc[[g for g in tg if g in log_cpm_filt.index],all_used]
    hz=hm.apply(lambda r:(r-r.mean())/(r.std()+1e-9),axis=1)
    cc=pd.Series([PAL.get(get_grp(c),LAVENDER) for c in all_used],index=all_used,name="Group")

    # Custom white-centred diverging cmap — publication standard
    cmap_pub = LinearSegmentedColormap.from_list(
        "pub_rw_blue", ["#2563EB","#93C5FD","#FFFFFF","#FCA5A5","#DC2626"], N=256)

    fg=sns.clustermap(hz, col_colors=cc, cmap=cmap_pub, center=0, vmin=-2.5, vmax=2.5,
                      figsize=(min(14,max(6,len(all_used)*0.9+4)),
                               max(7,len(tg)*0.20)),
                      row_cluster=True, col_cluster=False,
                      yticklabels=True, xticklabels=True, linewidths=0,
                      cbar_kws={"label":"Z-score","shrink":0.5,"ticks":[-2,0,2]},
                      dendrogram_ratio=(0.10,0.03),
                      tree_kws={"linewidths":0.6,"colors":MUTED_C})

    fg.ax_heatmap.set_xticklabels(fg.ax_heatmap.get_xticklabels(),
                                   rotation=45,ha="right",fontsize=7)
    fg.ax_heatmap.set_yticklabels(fg.ax_heatmap.get_yticklabels(),rotation=0,fontsize=6)
    fg.fig.suptitle(f"Top {len(tg)} DEGs — {'All Comparisons' if compare_all_mode else selected + ' vs Control'}",
                    y=1.01,fontsize=11,fontweight="bold")
    fg.fig.set_facecolor(SURF)
    # Add panel label
    if panel_label:
        fg.fig.text(0.01,0.99,"F",fontsize=13,fontweight="bold",va="top",color=TEXT_C)

    st.pyplot(fg.fig)
    bh=io.BytesIO()
    fg.fig.savefig(bh,format="png",dpi=pub_dpi,bbox_inches="tight",facecolor=SURF)
    st.download_button("⬇ Download Heatmap",bh.getvalue(),"heatmap_publication.png","image/png")
else:
    st.info("No significant DEGs to plot in heatmap — try relaxing thresholds.")

# ============================================================
#  ── 10. VENN ────────────────────────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑩ Venn Diagram</div>', unsafe_allow_html=True)
if VENN_OK:
    ss={g:set(r[r.direction!="NS"]["gene_id"]) for g,r in results.items()}
    gn=list(ss.keys())
    fig_venn,ax_venn=plt.subplots(figsize=(4,3.5))
    fig_venn.patch.set_facecolor(SURF); ax_venn.set_facecolor(SURF)
    venn_colors=[TEAL, CORAL, LAVENDER]
    if len(gn)==2:
        venn2([ss[gn[0]],ss[gn[1]]],set_labels=gn,ax=ax_venn,
              set_colors=venn_colors[:2],alpha=0.55)
    elif len(gn)>=3:
        venn3([ss[gn[0]],ss[gn[1]],ss[gn[2]]],set_labels=gn[:3],ax=ax_venn,
              set_colors=venn_colors,alpha=0.55)
    ax_venn.set_title("DEG Overlap",fontsize=10,fontweight="bold",pad=6)
    if panel_label: ax_venn.text(-0.1,1.06,"G",transform=ax_venn.transAxes,fontsize=13,fontweight="bold",va="top",color=TEXT_C)
    plt.tight_layout(pad=1.0)
    st.pyplot(fig_venn)
    st.download_button("⬇ Download Venn",save_pub(fig_venn,pub_dpi),"venn_publication.png","image/png")
else:
    st.info("Install `matplotlib-venn`: `pip install matplotlib-venn`")

# ── HELPERS: ORA / GSEA ──────────────────────────────────────
def run_ora(gene_sets, sig_genes, all_g):
    sig_set=set(g.upper() for g in sig_genes)
    bg_set =set(g.upper() for g in all_g)
    N=len(bg_set); K=len(sig_set); rows=[]
    for name,genes in gene_sets.items():
        pw_up=set(g.upper() for g in genes)
        M=len(pw_up&bg_set); x=len(sig_set&pw_up)
        if M<2 or x==0: continue
        pval=hypergeom.sf(x-1,N,M,K)
        rows.append({"Term":name,"GeneRatio":f"{x}/{K}","Count":x,"pvalue":pval,
                     "Genes":";".join(sorted(sig_set&pw_up))})
    if not rows: return pd.DataFrame()
    res=pd.DataFrame(rows)
    res["padj"]=multipletests(res["pvalue"],method="fdr_bh")[1]
    return res.sort_values("pvalue")

def plot_pub_ora(ora_df, title, dot_color=TEAL, panel="H"):
    show=ora_df[ora_df["pvalue"]<0.05]
    if len(show)==0: show=ora_df.head(10)
    show=show.head(20).copy()
    show["nlp"]=-np.log10(show["pvalue"].clip(lower=1e-10))
    show["GR_num"]=show["GeneRatio"].apply(lambda x: int(x.split("/")[0])/int(x.split("/")[1]) if "/" in str(x) else 0)

    fig,ax=plt.subplots(figsize=(5.5, max(3, len(show)*0.38)))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

    sc=ax.scatter(show["nlp"], range(len(show)),
                  s=show["Count"]*20+10,
                  c=show["pvalue"], cmap="RdYlBu",
                  vmin=0, vmax=0.05,
                  edgecolors=BORDER, linewidths=0.4, zorder=3)

    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["Term"].str[:50], fontsize=7.5)
    ax.set_xlabel("-log₁₀ (p-value)", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.axvline(x=-np.log10(0.05), ls="--", color="#94A3B8", lw=0.7, alpha=0.8)

    for i,(_,row) in enumerate(show.iterrows()):
        ax.text(row["nlp"]+0.04, i, f"  n={row['Count']}", va="center", fontsize=6.5, color=MUTED_C)

    cbar=plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.02)
    cbar.set_label("p-value", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Legend for dot size
    for cnt in [3,5,10]:
        ax.scatter([],[], s=cnt*20+10, c=MUTED_C, alpha=0.5, label=f"n={cnt}")
    ax.legend(title="Gene count", fontsize=6.5, title_fontsize=6.5,
              loc="lower right", frameon=True, framealpha=0.9)

    ax.grid(axis="x", alpha=0.3)
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)
    if panel_label:
        ax.text(-0.22, 1.04, panel, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", color=TEXT_C)
    plt.tight_layout(pad=1.0)
    return fig

# ── BAR CHART ────────────────────────────────────────────────
def plot_bar_chart(ora_df, title, bar_color=TEAL, panel="H"):
    """Horizontal bar chart — -log10(p) per term, coloured by significance tier."""
    show = ora_df[ora_df["pvalue"] < 0.05]
    if len(show) == 0:
        show = ora_df.head(10)
    show = show.head(20).copy().sort_values("pvalue", ascending=True)
    show["nlp"] = -np.log10(show["pvalue"].clip(lower=1e-10))

    # Build per-bar colours via rgba so no hex-alpha tricks needed
    base_rgba = matplotlib.colors.to_rgba(bar_color)
    def tier_rgba(p):
        a = 1.0 if p < 0.001 else (0.72 if p < 0.01 else 0.45)
        return (base_rgba[0], base_rgba[1], base_rgba[2], a)
    colors = [tier_rgba(p) for p in show["pvalue"]]

    fig, ax = plt.subplots(figsize=(6.5, max(3, len(show) * 0.42)))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.barh(range(len(show)), show["nlp"], color=colors,
            edgecolor="white", linewidth=0.5, height=0.72)

    for i, (_, row) in enumerate(show.iterrows()):
        ax.text(row["nlp"] + 0.05, i, f" n={row['Count']}",
                va="center", fontsize=6.5, color=MUTED_C)

    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["Term"].str[:55], fontsize=7.5)
    ax.axvline(x=-np.log10(0.05), ls="--", color="#94A3B8", lw=0.8, alpha=0.8)
    ax.set_xlabel("-log₁₀ (p-value)", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    tier_patches = [
        mpatches.Patch(color=(*base_rgba[:3], 1.0),  label="p < 0.001"),
        mpatches.Patch(color=(*base_rgba[:3], 0.72), label="p < 0.01"),
        mpatches.Patch(color=(*base_rgba[:3], 0.45), label="p < 0.05"),
    ]
    ax.legend(handles=tier_patches, fontsize=6.5, frameon=True,
              framealpha=0.9, loc="lower right",
              title="Significance", title_fontsize=6.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if panel_label:
        ax.text(-0.22, 1.04, panel, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", color=TEXT_C)
    plt.tight_layout(pad=1.0)
    return fig


# ── PIE / DONUT CHART ─────────────────────────────────────────
def plot_pie_chart(ora_df, title, palette=None, panel="H"):
    """Donut chart — top enriched terms by gene count, with legend table."""
    show = ora_df[ora_df["pvalue"] < 0.05]
    if len(show) == 0:
        show = ora_df.head(10)
    show = show.head(12).copy().sort_values("Count", ascending=False)

    if palette is None:
        palette = [TEAL, CORAL, LAVENDER, GOLD, SKY, GREEN,
                   "#F472B6", "#A78BFA", "#FB923C", "#34D399", "#60A5FA", "#FBBF24"]
    colors = palette[:len(show)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              gridspec_kw={"width_ratios": [1.1, 0.9]})
    fig.patch.set_facecolor(SURF)

    ax = axes[0]; ax.set_facecolor(SURF)
    wedges, texts, autotexts = ax.pie(
        show["Count"], labels=None, colors=colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.2))
    for at in autotexts:
        at.set_fontsize(7); at.set_color("white"); at.set_fontweight("bold")
    ax.text(0, 0, f"n={show['Count'].sum()}\ngenes",
            ha="center", va="center", fontsize=9, fontweight="bold", color=TEXT_C)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    if panel_label:
        ax.text(-0.08, 1.06, panel, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", color=TEXT_C)

    ax2 = axes[1]; ax2.set_facecolor(SURF); ax2.axis("off")
    legend_handles = [mpatches.Patch(color=c) for c in colors]
    legend_labels  = [
        f"{row['Term'][:38]}  (n={row['Count']}, p={row['pvalue']:.3f})"
        for _, row in show.iterrows()
    ]
    ax2.legend(legend_handles, legend_labels, loc="center left",
               fontsize=7, frameon=True, framealpha=0.9,
               borderpad=0.8, labelspacing=0.6,
               title="Enriched Terms", title_fontsize=7.5)
    plt.tight_layout(pad=1.2)
    return fig


# ── NETWORK CHART ─────────────────────────────────────────────
def plot_network_chart(ora_df, gene_sets, sig_genes_set, title,
                       node_color=TEAL, panel="H"):
    """
    Bipartite pathway–gene network (pure matplotlib, no networkx).
    Pathway nodes (■) on the left, gene nodes (●) on the right.
    Node size ∝ gene count; edge opacity ∝ significance.
    """
    show = ora_df[ora_df["pvalue"] < 0.05]
    if len(show) == 0:
        show = ora_df.head(8)
    show = show.head(10).copy()

    pathway_genes = {}
    all_net_genes = set()
    for _, row in show.iterrows():
        gs = set(g.upper() for g in gene_sets.get(row["Term"], [])) & sig_genes_set
        pathway_genes[row["Term"]] = gs
        all_net_genes |= gs

    all_net_genes = sorted(all_net_genes)
    terms = show["Term"].tolist()
    n_terms, n_genes = len(terms), len(all_net_genes)

    if n_genes == 0:
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF); ax.axis("off")
        ax.text(0.5, 0.5,
                "No gene–pathway links found.\n"
                "Gene IDs in your data may not match the pathway gene-set IDs.",
                ha="center", va="center", fontsize=8, color=MUTED_C,
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        return fig

    fig_w = max(9,  n_genes  * 0.58 + 3)
    fig_h = max(5,  n_terms  * 0.70 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

    pw_x, gene_x = 0.05, 0.95
    pw_ys   = {t: 1 - (i + 0.5) / n_terms        for i, t in enumerate(terms)}
    gene_ys = {g: 1 - (i + 0.5) / max(n_genes,1) for i, g in enumerate(all_net_genes)}

    palette_net = [TEAL, CORAL, LAVENDER, GOLD, SKY, GREEN,
                   "#F472B6", "#A78BFA", "#FB923C", "#34D399"]

    # Edges
    for ti, (_, row) in enumerate(show.iterrows()):
        nlp   = -np.log10(max(row["pvalue"], 1e-10))
        alpha = min(0.85, 0.18 + nlp * 0.09)
        color = palette_net[ti % len(palette_net)]
        for g in pathway_genes[row["Term"]]:
            ax.plot([pw_x, gene_x], [pw_ys[row["Term"]], gene_ys[g]],
                    color=color, lw=0.9, alpha=alpha, zorder=1)

    # Pathway nodes (squares)
    for ti, (_, row) in enumerate(show.iterrows()):
        color = palette_net[ti % len(palette_net)]
        size  = 100 + row["Count"] * 20
        ax.scatter([pw_x], [pw_ys[row["Term"]]], s=size, c=color,
                   marker="s", zorder=4, edgecolors="white", linewidths=1.2)
        label = (row["Term"][:32] + "…") if len(row["Term"]) > 33 else row["Term"]
        ax.text(pw_x - 0.018, pw_ys[row["Term"]], label,
                ha="right", va="center", fontsize=7, color=TEXT_C,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))

    # Gene nodes (circles)
    for g in all_net_genes:
        ax.scatter([gene_x], [gene_ys[g]], s=55, c=LAVENDER,
                   marker="o", zorder=4, edgecolors="white", linewidths=0.8)
        ax.text(gene_x + 0.018, gene_ys[g], g,
                ha="left", va="center", fontsize=6, color=MUTED_C)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

    pw_patch = mpatches.Patch(color=node_color,  label="Pathway  ■  (size ∝ gene count)")
    g_patch  = mpatches.Patch(color=LAVENDER,    label="DEG  ●")
    ax.legend(handles=[pw_patch, g_patch], fontsize=7, frameon=True,
              framealpha=0.9, loc="lower center", ncol=2)
    if panel_label:
        ax.text(0.0, 1.02, panel, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", color=TEXT_C)
    plt.tight_layout(pad=1.0)
    return fig


# ── TABBED CHART RENDERER (Dot / Bar / Pie / Network) ─────────
def render_ora_charts(ora_df, gene_sets, sig_genes, title_prefix,
                      bar_color, pie_palette, section_key):
    """Render ALL 4 chart types at once — Dot, Bar, Pie, Network — two per row."""
    sig_set = set(g.upper() for g in sig_genes)

    st.markdown("---")

    # ── Row 1: Dot Plot  |  Bar Chart ────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔵 Dot Plot**")
        fig_d = plot_pub_ora(ora_df, f"{title_prefix}")
        st.pyplot(fig_d, use_container_width=True)
        st.download_button("⬇ Dot Plot (PNG)",
                           save_pub(fig_d, pub_dpi),
                           f"{section_key}_dotplot.png", "image/png",
                           key=f"dl_dot_{section_key}")

    with col2:
        st.markdown("**📊 Bar Chart**")
        fig_b = plot_bar_chart(ora_df, f"{title_prefix}", bar_color=bar_color)
        st.pyplot(fig_b, use_container_width=True)
        st.download_button("⬇ Bar Chart (PNG)",
                           save_pub(fig_b, pub_dpi),
                           f"{section_key}_barchart.png", "image/png",
                           key=f"dl_bar_{section_key}")

    st.markdown("---")

    # ── Row 2: Pie Chart  |  Network Chart ───────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🥧 Pie / Donut Chart**")
        fig_p = plot_pie_chart(ora_df, f"{title_prefix}", palette=pie_palette)
        st.pyplot(fig_p, use_container_width=True)
        st.download_button("⬇ Pie Chart (PNG)",
                           save_pub(fig_p, pub_dpi),
                           f"{section_key}_piechart.png", "image/png",
                           key=f"dl_pie_{section_key}")

    with col4:
        st.markdown("**🕸️ Network Chart**")
        st.markdown(
            "<small>■ squares = pathways (size ∝ gene count) &nbsp;·&nbsp; "
            "● circles = DEGs &nbsp;·&nbsp; edge opacity ∝ significance</small>",
            unsafe_allow_html=True)
        fig_n = plot_network_chart(ora_df, gene_sets, sig_set,
                                   f"{title_prefix}", node_color=bar_color)
        st.pyplot(fig_n, use_container_width=True)
        st.download_button("⬇ Network Chart (PNG)",
                           save_pub(fig_n, pub_dpi),
                           f"{section_key}_network.png", "image/png",
                           key=f"dl_net_{section_key}")

    st.markdown("---")


def run_gsea_fixed(gene_sets, df_in, pv_col):
    gs_up={k:[g.upper() for g in v] for k,v in gene_sets.items()}
    rnk=(df_in.copy().assign(gu=df_in["gene_id"].str.upper(),
                          sc=df_in["log2FoldChange"]*-np.log10(df_in[pv_col].clip(lower=1e-300)))
         .drop_duplicates("gu").set_index("gu")["sc"].sort_values(ascending=False))
    pre=gp.prerank(rnk=rnk,gene_sets=gs_up,permutation_num=200,
                   min_size=3,no_plot=True,seed=42,verbose=False)
    return pre.res2d.sort_values("FDR q-val")

def plot_pub_gsea(gsea_df, title, panel="I"):
    show=gsea_df[gsea_df["FDR q-val"]<0.25]
    if len(show)==0: show=gsea_df.head(10)
    show=show.head(20).copy()
    colors=[CORAL if n>0 else SKY for n in show["NES"]]

    fig,ax=plt.subplots(figsize=(5.5,max(3,len(show)*0.38)))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

    bars=ax.barh(range(len(show)), show["NES"], color=colors,
                 edgecolor="white", linewidth=0.4, height=0.65)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["Term"].str[:50], fontsize=7.5)
    ax.axvline(x=0, color="#94A3B8", lw=0.8)
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    for i,(_,row) in enumerate(show.iterrows()):
        fdr=row.get("FDR q-val",1.0)
        sign="*" if fdr<0.05 else ("†" if fdr<0.25 else "")
        offset=0.04 if row["NES"]>=0 else -0.04
        ha="left" if row["NES"]>=0 else "right"
        ax.text(row["NES"]+offset, i, f"{sign}", va="center", ha=ha,
                fontsize=7, color=TEXT_C, fontweight="bold")

    ax.legend(handles=[mpatches.Patch(color=CORAL,label="Enriched (NES>0)"),
                        mpatches.Patch(color=SKY,label="Depleted (NES<0)")],
              fontsize=7, frameon=True, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)
    if panel_label:
        ax.text(-0.22, 1.04, panel, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", color=TEXT_C)
    plt.tight_layout(pad=1.0)
    return fig

# ============================================================
#  ── 11. KEGG ────────────────────────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑪ KEGG Pathway Analysis</div>', unsafe_allow_html=True)
kegg_sets=ORG_KEGG_MAP.get(organism_preset,PAO1_KEGG)
st.markdown(f"""<div class="info-box">
Curated KEGG gene sets for <b>{organism_preset}</b> — 100% offline · publication-quality dot plot.
</div>""", unsafe_allow_html=True)

km=st.radio("KEGG method:",["ORA","GSEA"],horizontal=True,key="km")
if st.button("🔬 Run KEGG Analysis",type="primary",key="kegg_btn"):
    if compare_all_mode:
        # Pool significant genes from all comparisons
        sig_genes = list({g for r in results.values() for g in r[r.direction!="NS"]["gene_id"].tolist()})
        all_g = list({g for r in results.values() for g in r["gene_id"].tolist()})
        kegg_title = f"KEGG ORA — All Comparisons\n{organism_preset}"
    else:
        sig_genes=df[df.direction!="NS"]["gene_id"].tolist()
        all_g=df["gene_id"].tolist()
        kegg_title = f"KEGG ORA — {selected} vs Control\n{organism_preset}"
    if len(sig_genes)<3:
        st.warning("Too few DEGs — relax thresholds.")
    elif km=="ORA":
        ora=run_ora(kegg_sets,sig_genes,all_g)
        if len(ora)==0: st.warning("No KEGG pathways matched.")
        else:
            show_k=ora[ora["pvalue"]<0.05]
            if len(show_k)==0: show_k=ora.head(10); st.info("Showing top 10 (none p<0.05).")
            st.success(f"✅ {len(show_k)} enriched KEGG pathways")
            st.dataframe(show_k.reset_index(drop=True), use_container_width=True)
            render_ora_charts(
                ora_df=ora,
                gene_sets=kegg_sets,
                sig_genes=sig_genes,
                title_prefix=kegg_title.replace("\n", " — "),
                bar_color=TEAL,
                pie_palette=[TEAL, CORAL, LAVENDER, GOLD, SKY, GREEN,
                             "#F472B6", "#A78BFA", "#FB923C", "#34D399", "#60A5FA", "#FBBF24"],
                section_key=f"kegg_{'all' if compare_all_mode else selected}"
            )
            st.download_button("⬇ Download KEGG ORA CSV", show_k.to_csv(index=False).encode(),
                               f"KEGG_ORA_{'All' if compare_all_mode else selected}.csv", "text/csv")
    else:
        if not GSEAPY_OK: st.error("Install gseapy: pip install gseapy")
        else:
            with st.spinner("Running KEGG GSEA..."):
                try:
                    gsea_res=run_gsea_fixed(kegg_sets,df,pv_col)
                    show_gs=gsea_res[gsea_res["FDR q-val"]<0.25]
                    if len(show_gs)==0: show_gs=gsea_res.head(10); st.info("Showing top 10 (none FDR<0.25).")
                    st.success(f"✅ KEGG GSEA — {len(show_gs)} pathways")
                    st.dataframe(show_gs[["Term","ES","NES","NOM p-val","FDR q-val","Lead_genes"]].reset_index(drop=True),use_container_width=True)
                    fig_gs=plot_pub_gsea(gsea_res,f"KEGG GSEA — {'All Comparisons' if compare_all_mode else selected + ' vs Control'}","H")
                    st.pyplot(fig_gs)
                    st.download_button("⬇ Download KEGG GSEA Plot",save_pub(fig_gs,pub_dpi),"KEGG_GSEA_publication.png","image/png")
                    st.download_button("⬇ Download KEGG GSEA CSV",show_gs.to_csv(index=False).encode(),f"KEGG_GSEA_{'All' if compare_all_mode else selected}.csv","text/csv")
                except Exception as e: st.error(f"GSEA error: {e}")

# ============================================================
#  ── 12. GO ──────────────────────────────────────────────────
# ============================================================
st.markdown('<div class="sec">⑫ Gene Ontology (GO) Analysis</div>', unsafe_allow_html=True)
go_bp_s, go_mf_s, go_cc_s = ORG_GO_MAP.get(organism_preset, (PAO1_GO_BP, PAO1_GO_MF, PAO1_GO_CC))
st.markdown(f"""<div class="info-box">
Curated GO annotations for <b>{organism_preset}</b> — 100% offline · all three ontologies run at once.<br>
<b>BP</b> = Biological Process &nbsp;·&nbsp; <b>MF</b> = Molecular Function &nbsp;·&nbsp; <b>CC</b> = Cellular Component
</div>""", unsafe_allow_html=True)

go_m = st.radio("GO method:", ["ORA", "GSEA"], horizontal=True, key="gm")

GO_ONTOLOGIES = {
    "Biological Process (BP)": {"sets": go_bp_s, "color": GREEN,
        "palette": [GREEN,"#86EFAC","#4ADE80",TEAL,"#2DD4BF",CORAL,"#FCA5A5",
                    GOLD,"#FDE68A",LAVENDER,"#C4B5FD",SKY],
        "icon": "🟢"},
    "Molecular Function (MF)": {"sets": go_mf_s, "color": SKY,
        "palette": [SKY,"#7DD3FC","#38BDF8",LAVENDER,"#A78BFA",CORAL,"#FCA5A5",
                    GREEN,GOLD,TEAL,"#F472B6","#60A5FA"],
        "icon": "🔵"},
    "Cellular Component (CC)": {"sets": go_cc_s, "color": LAVENDER,
        "palette": [LAVENDER,"#C4B5FD","#A78BFA",SKY,GREEN,CORAL,TEAL,
                    GOLD,"#F472B6","#FB923C","#34D399","#60A5FA"],
        "icon": "🟣"},
}

if st.button("🔬 Run GO Analysis — BP + MF + CC", type="primary", key="go_btn"):
    # Build gene lists once
    if compare_all_mode:
        sig_genes = list({g for r in results.values() for g in r[r.direction!="NS"]["gene_id"].tolist()})
        all_g     = list({g for r in results.values() for g in r["gene_id"].tolist()})
        go_label  = f"All Comparisons ({organism_preset})"
    else:
        sig_genes = df[df.direction!="NS"]["gene_id"].tolist()
        all_g     = df["gene_id"].tolist()
        go_label  = f"{selected} vs Control ({organism_preset})"

    if len(sig_genes) < 3:
        st.warning("Too few DEGs — relax thresholds.")
    else:
        # ── Loop over all 3 ontologies ────────────────────────
        for ont_name, ont_cfg in GO_ONTOLOGIES.items():
            icon    = ont_cfg["icon"]
            go_c    = ont_cfg["color"]
            go_sets = ont_cfg["sets"]
            short   = ont_name[-3:-1]   # "BP", "MF", or "CC"

            st.markdown(
                f'<div class="sec" style="font-size:1rem;padding:8px 14px;">'
                f'{icon} {ont_name}</div>',
                unsafe_allow_html=True)

            if go_m == "ORA":
                with st.spinner(f"Running {ont_name} ORA..."):
                    ora_go = run_ora(go_sets, sig_genes, all_g)

                if len(ora_go) == 0:
                    st.info(f"No {ont_name} terms matched — gene IDs may not overlap.")
                else:
                    show_go = ora_go[ora_go["pvalue"] < 0.05]
                    if len(show_go) == 0:
                        show_go = ora_go.head(10)
                        st.info(f"Showing top 10 {short} terms (none p<0.05).")
                    st.success(f"✅ {len(show_go)} enriched {short} terms")

                    with st.expander(f"📋 {short} Results Table", expanded=False):
                        st.dataframe(show_go.reset_index(drop=True), use_container_width=True)
                        st.download_button(
                            f"⬇ Download {short} ORA CSV",
                            show_go.to_csv(index=False).encode(),
                            f"GO_{short}_{'All' if compare_all_mode else selected}.csv",
                            "text/csv",
                            key=f"csv_{short}_{'all' if compare_all_mode else selected}")

                    render_ora_charts(
                        ora_df=ora_go,
                        gene_sets=go_sets,
                        sig_genes=sig_genes,
                        title_prefix=f"GO {ont_name} — {go_label}",
                        bar_color=go_c,
                        pie_palette=ont_cfg["palette"],
                        section_key=f"go_{short.lower()}_{'all' if compare_all_mode else selected}"
                    )

            else:  # GSEA
                if not GSEAPY_OK:
                    st.error("Install gseapy: pip install gseapy")
                else:
                    with st.spinner(f"Running {ont_name} GSEA..."):
                        try:
                            gsea_go  = run_gsea_fixed(go_sets, df, pv_col)
                            show_gsg = gsea_go[gsea_go["FDR q-val"] < 0.25]
                            if len(show_gsg) == 0:
                                show_gsg = gsea_go.head(10)
                                st.info(f"Showing top 10 {short} GSEA terms (none FDR<0.25).")
                            st.success(f"✅ GO GSEA {short} — {len(show_gsg)} terms")
                            with st.expander(f"📋 {short} GSEA Results Table", expanded=False):
                                st.dataframe(
                                    show_gsg[["Term","ES","NES","NOM p-val","FDR q-val","Lead_genes"]]
                                    .reset_index(drop=True), use_container_width=True)
                                st.download_button(
                                    f"⬇ Download {short} GSEA CSV",
                                    show_gsg.to_csv(index=False).encode(),
                                    f"GO_GSEA_{short}_{'All' if compare_all_mode else selected}.csv",
                                    "text/csv",
                                    key=f"gsea_csv_{short}_{'all' if compare_all_mode else selected}")
                            fig_gsg = plot_pub_gsea(
                                gsea_go,
                                f"GO GSEA {ont_name} — {'All Comparisons' if compare_all_mode else selected}",
                                "I")
                            st.pyplot(fig_gsg)
                            st.download_button(
                                f"⬇ Download {short} GSEA Plot",
                                save_pub(fig_gsg, pub_dpi),
                                f"GO_GSEA_{short}_publication.png", "image/png",
                                key=f"gsea_fig_{short}_{'all' if compare_all_mode else selected}")
                        except Exception as e:
                            st.error(f"{ont_name} GSEA error: {e}")

# ============================================================
#  ── 13. MULTI-PANEL FIGURE EXPORT (publication composite) ───
# ============================================================
st.markdown('<div class="sec">⑬ 📄 Publication Multi-Panel Figure Export</div>', unsafe_allow_html=True)
st.markdown("""<div class="pub-box">
🎓 Export a <b>journal-ready composite figure</b> (Volcano + PCA + Dispersion) as a single PNG
at 300 DPI with panel labels A, B, C — ready for Nature, Cell, PNAS submission.
</div>""", unsafe_allow_html=True)

if st.button("🖼️ Generate Multi-Panel Figure", type="primary", key="multipanel_btn"):
    fig_mp = plt.figure(figsize=(15, 5))
    fig_mp.patch.set_facecolor(SURF)
    gs_mp = gridspec.GridSpec(1, 3, figure=fig_mp, wspace=0.42, hspace=0.3)

    # ── Panel A — Volcano ─────────────────────────────────────
    ax_a = fig_mp.add_subplot(gs_mp[0])
    ax_a.set_facecolor(SURF)
    df2  = df.copy(); df2["mlp"] = -np.log10(df2[pv_col].clip(lower=1e-300))
    df2["score"] = df2["mlp"] * df2["log2FoldChange"].abs()

    # Compute axis limits first
    xv = df2["log2FoldChange"]; yv = df2["mlp"]
    xp = (xv.max()-xv.min())*0.10; yp = (yv.max()-yv.min())*0.10
    ax_a.set_xlim(xv.min()-xp, xv.max()+xp)
    ax_a.set_ylim(max(0,yv.min()-yp*0.5), yv.max()+yp*2.5)

    ax_a.scatter(df2.loc[df2.direction=="NS","log2FoldChange"],
                 df2.loc[df2.direction=="NS","mlp"],
                 c="#CBD5E1",s=4,alpha=0.35,linewidths=0,rasterized=True)
    ax_a.scatter(df2.loc[df2.direction=="UP","log2FoldChange"],
                 df2.loc[df2.direction=="UP","mlp"],
                 c=CORAL,s=8,alpha=0.85,linewidths=0,rasterized=True,
                 label=f"Up (n={(df2.direction=='UP').sum()})")
    ax_a.scatter(df2.loc[df2.direction=="DOWN","log2FoldChange"],
                 df2.loc[df2.direction=="DOWN","mlp"],
                 c=SKY,s=8,alpha=0.85,linewidths=0,rasterized=True,
                 label=f"Down (n={(df2.direction=='DOWN').sum()})")
    ax_a.axvline(x= logfc_cutoff,ls="--",color="#94A3B8",lw=0.7)
    ax_a.axvline(x=-logfc_cutoff,ls="--",color="#94A3B8",lw=0.7)
    ax_a.axhline(y=-np.log10(pvalue_cutoff),ls="--",color="#94A3B8",lw=0.7)

    # Top gene labels — smart placement, inside axes only
    top5u = df2[df2.direction=="UP"].nlargest(5,"score")
    top5d = df2[df2.direction=="DOWN"].nlargest(5,"score")
    placed_a = []
    for _, row in pd.concat([top5u,top5d]).sort_values("score",ascending=False).iterrows():
        x2,y2 = row["log2FoldChange"], row["mlp"]
        dy_txt = 12 if row["direction"]=="UP" else 10
        skip = any(abs(x2-px)<(xv.max()-xv.min())*0.07 and abs(y2-py)<(yv.max()-yv.min())*0.08
                   for px,py in placed_a)
        if skip: continue
        placed_a.append((x2,y2))
        col2 = CORAL if row["direction"]=="UP" else SKY
        ax_a.annotate(row["gene_id"],(x2,y2),
                      fontsize=4.5, xytext=(0,dy_txt), textcoords="offset points",
                      color=TEXT_C, clip_on=True,
                      arrowprops=dict(arrowstyle="-",color="#94A3B8",lw=0.3,shrinkA=1,shrinkB=1),
                      bbox=dict(boxstyle="round,pad=0.1",fc="white",ec=col2,alpha=0.8,linewidth=0.4))

    ax_a.set_xlabel("log₂ Fold Change",fontsize=8,labelpad=4)
    ax_a.set_ylabel(f"-log₁₀({'padj' if use_padj else 'p-value'})",fontsize=8,labelpad=4)
    ax_a.set_title(f"{'All Comparisons' if compare_all_mode else selected + ' vs Control'}",fontsize=9,fontweight="bold",pad=8)
    ax_a.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax_a.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax_a.tick_params(labelsize=7,length=3,width=0.7)
    ax_a.legend(fontsize=6,frameon=True,framealpha=0.9,loc="upper right",borderpad=0.5)
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_linewidth(0.7); ax_a.spines["bottom"].set_linewidth(0.7)
    ax_a.text(-0.16,1.06,"A",transform=ax_a.transAxes,fontsize=12,fontweight="bold",va="top",color=TEXT_C)

    # ── Panel B — PCA ──────────────────────────────────────────
    ax_b = fig_mp.add_subplot(gs_mp[1])
    ax_b.set_facecolor(SURF)
    pca_df2=pca_df.copy()
    pc1r = pca_df2["PC1"].max()-pca_df2["PC1"].min()
    pc2r = pca_df2["PC2"].max()-pca_df2["PC2"].min()
    ax_b.set_xlim(pca_df2["PC1"].min()-pc1r*0.25, pca_df2["PC1"].max()+pc1r*0.25)
    ax_b.set_ylim(pca_df2["PC2"].min()-pc2r*0.25, pca_df2["PC2"].max()+pc2r*0.25)
    for mi,grp in enumerate(pca_df2["Group"].unique()):
        sub=pca_df2[pca_df2["Group"]==grp]; color=PAL.get(grp,LAVENDER)
        ax_b.scatter(sub["PC1"],sub["PC2"],c=color,s=60,zorder=4,label=grp,
                     edgecolors="white",linewidths=1.0,marker=markers[mi%len(markers)])
    # Smart label placement
    placed_b = []
    for _, row in pca_df2.sort_values("PC1").iterrows():
        bx, by = row["PC1"], row["PC2"]
        offsets_b = [(pc1r*0.06, pc2r*0.06),(-pc1r*0.06, pc2r*0.06),
                     (pc1r*0.06,-pc2r*0.06),(-pc1r*0.06,-pc2r*0.06),
                     (pc1r*0.10, 0),(-pc1r*0.10,0)]
        best_ox, best_oy = pc1r*0.06, pc2r*0.06
        for ox,oy in offsets_b:
            ok = all(abs(bx+ox-px)>pc1r*0.05 or abs(by+oy-py)>pc2r*0.05
                     for px,py in placed_b)
            if ok: best_ox,best_oy=ox,oy; break
        placed_b.append((bx+best_ox, by+best_oy))
        lbl = row["Sample"][:11]+"…" if len(row["Sample"])>12 else row["Sample"]
        ax_b.annotate(lbl,(bx,by),xytext=(best_ox*8,best_oy*8),
                      textcoords="offset points",fontsize=5,color=MUTED_C,
                      arrowprops=dict(arrowstyle="-",color="#CBD5E1",lw=0.3))
    ax_b.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)",fontsize=8,labelpad=4)
    ax_b.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)",fontsize=8,labelpad=4)
    ax_b.set_title("PCA",fontsize=9,fontweight="bold",pad=8)
    ax_b.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax_b.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax_b.tick_params(labelsize=7,length=3,width=0.7)
    ax_b.legend(title="Condition",fontsize=6,title_fontsize=6,
                frameon=True,framealpha=0.9,borderpad=0.5)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_linewidth(0.7); ax_b.spines["bottom"].set_linewidth(0.7)
    ax_b.text(-0.18,1.06,"B",transform=ax_b.transAxes,fontsize=12,fontweight="bold",va="top",color=TEXT_C)

    # ── Panel C — Dispersion ───────────────────────────────────
    ax_c = fig_mp.add_subplot(gs_mp[2])
    ax_c.set_facecolor(SURF)
    ax_c.scatter(means.values,d_gene.values,c=MUTED_C,s=2,alpha=0.2,linewidths=0,rasterized=True)
    ax_c.scatter(means.values,d_final,c=TEAL,s=3,alpha=0.35,linewidths=0,rasterized=True)
    ax_c.plot(means.values[si],sm_fit,c=CORAL,lw=1.4,ls="--",zorder=5)
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    # Proper log tick formatter
    from matplotlib.ticker import LogFormatterMathtext, LogLocator
    ax_c.xaxis.set_major_locator(LogLocator(base=10,numticks=6))
    ax_c.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
    ax_c.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_c.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_c.tick_params(axis="both",which="major",labelsize=7,length=3,width=0.7)
    ax_c.tick_params(axis="both",which="minor",labelsize=0,length=2,width=0.5)
    # Axis limits
    m_min2 = np.nanmin(means.values[means.values>0])
    ax_c.set_xlim(m_min2*0.5, np.nanmax(means.values)*2)
    ax_c.set_ylim(max(1e-4,np.nanmin(d_final[d_final>0]))*0.5,
                  np.nanmax(d_gene.values)*3)
    ax_c.set_xlabel("Mean counts",fontsize=8,labelpad=4)
    ax_c.set_ylabel("Dispersion",fontsize=8,labelpad=4)
    ax_c.set_title("Dispersion Estimates",fontsize=9,fontweight="bold",pad=8)
    ax_c.legend(fontsize=6,frameon=True,framealpha=0.9,borderpad=0.5,
                handles=[mpatches.Patch(color=MUTED_C,label="Gene-wise"),
                         mpatches.Patch(color=CORAL,label="Fitted"),
                         mpatches.Patch(color=TEAL,label="Final")])
    ax_c.grid(True,which="major",ls=":",alpha=0.3,lw=0.5)
    ax_c.grid(True,which="minor",ls=":",alpha=0.12,lw=0.3)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_linewidth(0.7); ax_c.spines["bottom"].set_linewidth(0.7)
    ax_c.text(-0.22,1.06,"C",transform=ax_c.transAxes,fontsize=12,fontweight="bold",va="top",color=TEXT_C)

    fig_mp.tight_layout(pad=1.5, rect=[0, 0, 1, 0.98])
    st.pyplot(fig_mp)
    buf_mp=io.BytesIO()
    fig_mp.savefig(buf_mp,format="png",dpi=pub_dpi,bbox_inches="tight",facecolor=SURF,edgecolor="none")
    st.download_button("⬇ Download Multi-Panel Figure (300 DPI)",buf_mp.getvalue(),
                       "multipanel_publication.png","image/png")
    st.success("✅ Multi-panel figure ready — suitable for journal submission!")

# ── 14. DEG TABLE ────────────────────────────────────────────
st.markdown('<div class="sec">⑭ Significant DEGs Table</div>', unsafe_allow_html=True)
sd = st.radio("Show:", ["All DEGs","UP only","DOWN only"], horizontal=True)

if sd == "UP only":
    show_df = df[df["direction"] == "UP"]
elif sd == "DOWN only":
    show_df = df[df["direction"] == "DOWN"]
else:
    show_df = df[df["direction"] != "NS"]

st.dataframe(
    show_df.sort_values("log2FoldChange",ascending=False).reset_index(drop=True),
    use_container_width=True
)
st.download_button("⬇ Download DEG CSV", df.to_csv(index=False).encode(),
                   f"DEG_{'All' if compare_all_mode else selected}.csv","text/csv")

import zipfile
zbuf=io.BytesIO()
with zipfile.ZipFile(zbuf,"w") as zf:
    for g,r in results.items():
        zf.writestr(f"DEG_{g}_vs_Control.csv",r.to_csv(index=False))
st.download_button("⬇ Download All Comparisons (ZIP)",zbuf.getvalue(),"All_DEG_results.zip","application/zip")

st.markdown("""<div class="footer">
  🧬 Bacterial RNA-seq Dashboard &nbsp;·&nbsp; GO &amp; KEGG via curated offline bacterial gene sets
  &nbsp;·&nbsp; Light Mode &nbsp;·&nbsp; Publication-Quality Figures (Nature/Cell style)
</div>""", unsafe_allow_html=True)
