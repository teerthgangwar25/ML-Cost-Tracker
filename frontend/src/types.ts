export interface MLRun {
  run_id?: string;
  experiment_id?: string;
  n_estimators: number | null;
  max_depth: number | null;
  min_samples_split: number | null;
  accuracy: number;
  train_duration_sec: number;
  estimated_cost_usd: number;
  is_pareto_optimal: boolean;
  source?: string;
  trial_number?: number;
  [key: string]: any;
}

export interface ShapImportance {
  Hyperparameter: string;
  Mean_Absolute_SHAP: number;
}

export interface OptunaSummary {
  study_name?: string;
  n_trials?: number;
  best_value?: number;
  best_params?: any;
}

export interface DashboardData {
  runs: MLRun[];
  shap_importance: ShapImportance[];
  optuna_summary?: OptunaSummary;
}

export interface ConfigData {
  instance_types: string[];
}

export interface ApiResponse {
  status: string;
  data?: DashboardData;
  message?: string;
}

export interface ConfigResponse {
  status: string;
  data?: ConfigData;
  message?: string;
}
