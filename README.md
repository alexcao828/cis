# Classifier-induced stopping (CIS)
Base code for 'A Policy for Early Sequence Classification' by Alexander Cao, Jean Utke, and Diego Klabjan (ICANN 2023)

Three top-level folders correpsond with the three experiments from the paper. Please see the paper for data sources. Within each experiment's folder:
-  data folder: code for extracting features from raw data and splitting into train, validation, and test sets
-  ppo: code for PPO benchmark (only $\mu=10^{-3}$ for simplicity)
-  larm: code for LARM benchmark ($\mu=10^{-3}$)
-  cis: code for proposed CIS method ($\mu=10^{-3}$)
-  paretoAUC: code for plotting Pareto frontiers of above three methods (multiple mu) and calculating AUC
-  plotEx (for imdb), TbyClass (ecg), plotAnalyze (stockOption): code for plotting Figures 3, 4 (right), and 6 respectively in the paper i.e. human interpretation of results
