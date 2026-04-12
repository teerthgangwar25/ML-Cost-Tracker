import { useEffect, useState, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, BarChart, Bar, XAxis as BarXAxis, YAxis as BarYAxis } from 'recharts';
import { LayoutDashboard, FlaskConical, Database, Cpu, Download, Bell, CircleUserRound, Activity } from 'lucide-react';
import type { DashboardData, MLRun, ConfigData } from './types';
import './index.css';

function App() {
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [data, setData] = useState<DashboardData | null>(null);
  
  // State for controls
  const [source, setSource] = useState("MLflow (Grid Search)");
  const [instanceType, setInstanceType] = useState("local_cpu");
  
  const [loading, setLoading] = useState(true);
  const [budget, setBudget] = useState(150000);

  // Load configuration once
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/config')
      .then(res => res.json())
      .then(res => {
        if (res.status === 'success') {
          setConfig(res.data);
          if (res.data.instance_types && res.data.instance_types.length > 0) {
            setInstanceType(res.data.instance_types[0]);
          }
        }
      })
      .catch(err => console.error("Config fetch error:", err));
  }, []);

  // Fetch dashboard data whenever controls change
  useEffect(() => {
    setLoading(true);
    const url = `http://127.0.0.1:8000/api/dashboard-data?source=${encodeURIComponent(source)}&instance_type=${encodeURIComponent(instanceType)}`;
    fetch(url)
      .then(res => res.json())
      .then(res => {
        if (res.status === 'success') {
          setData(res.data);
          
          // Auto-set the budget near the 75th percentile of optimal cost
          const pRuns = res.data.runs.filter((r: MLRun) => r.is_pareto_optimal);
          if (pRuns.length > 0) {
            const costs = pRuns.map((r: MLRun) => r.estimated_cost_usd).sort((a: number, b: number) => a - b);
            const q3 = costs[Math.floor(costs.length * 0.75)];
            setBudget(Math.ceil(q3));
          } else {
            setBudget(10);
          }
        } else {
          // Reset data on error to show empty state
          setData(null);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Data fetch error:", err);
        setLoading(false);
      });
  }, [source, instanceType]);

  const { runs = [], shap_importance = [], optuna_summary = {} } = data || {};

  // Find the optimal run within budget
  const optimalRun = useMemo(() => {
    if (!runs || runs.length === 0) return null;
    const affordableParetoRuns = runs.filter(
      r => r.estimated_cost_usd <= budget && r.is_pareto_optimal
    );
    return affordableParetoRuns.sort((a, b) => b.accuracy - a.accuracy)[0] || runs[0];
  }, [runs, budget]);

  const paretoLineData = useMemo(() => {
    return runs.filter(r => r.is_pareto_optimal)
               .sort((a, b) => a.estimated_cost_usd - b.estimated_cost_usd);
  }, [runs]);

  // Optuna convergence calculation
  const convergenceData = useMemo(() => {
    if (!source.includes("Optuna") || !runs.length) return [];
    
    // Sort optuna trials by trial number
    const optunaTrials = runs.filter(r => r.source === 'Optuna (TPE)' && r.trial_number !== undefined)
                             .sort((a, b) => (a.trial_number || 0) - (b.trial_number || 0));
    
    let maxSoFar = 0;
    return optunaTrials.map(trial => {
      maxSoFar = Math.max(maxSoFar, trial.accuracy);
      return {
        ...trial,
        best_so_far: maxSoFar
      };
    });
  }, [runs, source]);

  // Download CSV
  const handleDownloadCsv = () => {
    if (!runs.length) return;
    const cols = ['run_id', 'source', 'accuracy', 'estimated_cost_usd', 'n_estimators', 'max_depth', 'min_samples_split', 'duration_hours', 'is_pareto_optimal'];
    
    // Build CSV string
    const csvRows = [];
    csvRows.push(cols.join(',')); // Header
    
    for (const row of runs) {
      const values = cols.map(col => {
        const val = row[col];
        return val === null || val === undefined ? '' : String(val);
      });
      csvRows.push(values.join(','));
    }
    
    const csvData = csvRows.join('\n');
    const blob = new Blob([csvData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', 'ml_experiments.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="flex h-screen bg-[var(--color-dark-bg)] text-slate-300 font-sans overflow-hidden">
      {/* Sidebar Controls */}
      <aside className="w-56 xl:w-72 bg-[#111827] border-r border-slate-800 flex flex-col p-4 shadow-xl z-20 overflow-y-auto hidden md:flex transition-all duration-300">
        <div className="flex items-center gap-3 mb-8 px-2 mt-2">
          <div className="w-8 h-8 rounded bg-blue-600 flex items-center justify-center text-white shadow-[0_0_10px_rgba(37,99,235,0.5)]">
            <LayoutDashboard size={18} />
          </div>
          <h1 className="text-white font-[600] tracking-tight text-sm xl:text-base">ML COST TRACKER</h1>
        </div>
        
        <div className="px-2 mb-6">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Master Controls</h3>
            
            {/* Source Selector */}
            <div className="mb-5">
               <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 mb-2">
                  <Database size={14} className="text-blue-400"/> Data Source
               </label>
               <select 
                  value={source} 
                  onChange={e => setSource(e.target.value)}
                  className="w-full bg-[#0f172a] border border-slate-700 text-slate-300 text-sm rounded-lg p-2.5 focus:ring-blue-500 focus:border-blue-500 appearance-none outline-none"
                >
                  <option>MLflow (Grid Search)</option>
                  <option>Optuna (TPE)</option>
                  <option>Both Combined</option>
               </select>
            </div>

            {/* Instance Selector */}
            <div className="mb-5">
               <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 mb-2 mt-4 mt-2">
                  <Cpu size={14} className="text-blue-400"/> Instance Cost Type
               </label>
               <select 
                  value={instanceType} 
                  onChange={e => setInstanceType(e.target.value)}
                  className="w-full bg-[#0f172a] border border-slate-700 text-slate-300 text-sm rounded-lg p-2.5 focus:ring-blue-500 focus:border-blue-500 appearance-none outline-none"
                >
                  {config?.instance_types?.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
               </select>
               <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
                   Affects Optuna cost tracking dynamically. MLflow costs are locked to computation generation.
               </p>
            </div>
        </div>

        <nav className="flex-1 space-y-1">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-2 mb-3 mt-4">Navigation</h3>
          <a href="#" className="flex items-center gap-3 px-3 py-2 bg-slate-800/80 text-white rounded-lg transition-colors border-l-2 border-blue-500">
            <LayoutDashboard size={16} className="text-blue-400" /> <span className="font-medium text-sm">Dashboard</span>
          </a>
          <a href="#" className="flex items-center gap-3 px-3 py-2 hover:bg-slate-800/50 text-slate-400 hover:text-white rounded-lg transition-colors">
            <FlaskConical size={16} /> <span className="font-medium text-sm">Experiments</span>
          </a>
        </nav>
        
        <div className="mt-auto pt-4 border-t border-slate-800 flex items-center gap-2 px-3 text-slate-500 text-xs">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div> Backend Synced
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-y-auto bg-[var(--color-dark-bg)] bg-gradient-to-br from-[#0B111D] to-[#0f172a]/50 relative">
        
        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 z-50 bg-[#0B111D]/80 backdrop-blur-sm flex items-center justify-center">
             <div className="text-blue-500 font-bold tracking-widest uppercase flex items-center gap-3">
                 <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div> Re-calculating Model Space...
             </div>
          </div>
        )}

        {/* Header */}
        <header className="px-6 py-5 lg:px-8 flex justify-between items-center z-10 sticky top-0 bg-[#0B111D]/90 backdrop-blur-md border-b border-slate-800/80 shadow-sm">
          <div className="flex items-center gap-4 xl:gap-6">
            <h2 className="text-lg xl:text-xl font-[700] text-white tracking-wide uppercase">Performance Dashboard</h2>
            {optuna_summary?.n_trials && (
               <span className="hidden lg:inline-flex px-2 py-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs rounded-full">
                  Optuna Active ({optuna_summary.n_trials} Trials)
               </span>
            )}
          </div>
          <div className="flex items-center gap-3 xl:gap-5 text-slate-400">
            <div className="cursor-pointer hover:bg-slate-800 p-2 rounded-full transition-colors relative">
              <Bell size={18} className="hover:text-white" />
              <span className="absolute top-1.5 right-2 w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
            </div>
            <div className="flex items-center gap-2 hover:bg-slate-800 p-1 pr-3 rounded-full transition-colors cursor-pointer border border-slate-800">
              <CircleUserRound size={28} className="text-blue-400" />
              <span className="text-sm font-medium text-slate-200 hidden xl:block">Teerth</span>
            </div>
          </div>
        </header>

        {(!data || runs.length === 0) && !loading ? (
             <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-8">
                <Database size={48} className="mb-4 text-slate-600"/>
                <h3 className="text-xl font-bold text-slate-300 mb-2">No Data Found</h3>
                <p>Ensure `generate_optuna_runs.py` or `feature_builder.py` have been executed for <strong>{source}</strong>.</p>
             </div>
        ) : (
          <div className="p-6 lg:p-8 space-y-6">
            
            {/* Top Grid: Main visuals */}
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 xl:gap-8 min-h-[500px]">
              
              {/* SHAP Chart - Col 1 */}
              <section className="col-span-1 bg-[var(--color-glass-bg)] flex flex-col justify-between backdrop-blur rounded-xl border border-[var(--color-glass-border)] p-5 shadow-[0_4px_30px_rgba(0,0,0,0.3)]">
                <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase">Feature Importance (SHAP)</h3>
                </div>
                {shap_importance.length > 0 ? (
                    <div className="flex-1 min-h-[300px]">
                     <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={shap_importance} layout="vertical" margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#1e293b" />
                        <BarXAxis type="number" stroke="#64748b" fontSize={11} tickFormatter={(val) => val.toFixed(2)} />
                        <BarYAxis dataKey="Hyperparameter" type="category" stroke="#94a3b8" fontSize={11} width={85} />
                        <RechartsTooltip cursor={{ fill: 'rgba(255,255,255,0.02)' }} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }} />
                        <Bar dataKey="Mean_Absolute_SHAP" fill="url(#blue-gradient)" radius={[0, 4, 4, 0]} barSize={20} />
                        <defs>
                            <linearGradient id="blue-gradient" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stopColor="#2563EB" />
                            <stop offset="100%" stopColor="#60A5FA" />
                            </linearGradient>
                        </defs>
                        </BarChart>
                     </ResponsiveContainer>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center text-xs text-slate-500 p-4text-center">Not enough parameters for reliable SHAP explanation in this subset.</div>
                )}
              </section>

              {/* Middle Column (Chart + Budget) - Col 2&3 */}
              <div className="col-span-1 xl:col-span-2 flex flex-col gap-6 h-full">
                {/* Pareto Chart */}
                <section className="flex-[2] bg-[var(--color-glass-bg)] backdrop-blur rounded-xl border border-[var(--color-glass-border)] p-5 shadow-[0_4px_30px_rgba(0,0,0,0.3)] relative min-h-[350px]">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase">Pareto Front: Cost vs. Accuracy</h3>
                    <div className="flex items-center gap-3">
                        {source.includes('Both') && (
                            <div className="hidden sm:flex text-[10px] items-center gap-3 mr-2">
                                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-slate-500"></span> MLflow</span>
                                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-emerald-500"></span> Optuna</span>
                            </div>
                        )}
                        <span className="px-2 py-1 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] uppercase tracking-wide rounded">Interactive</span>
                    </div>
                  </div>
                  <ResponsiveContainer width="100%" height="88%">
                    <ScatterChart margin={{ top: 10, right: 30, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="4 4" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="estimated_cost_usd" type="number" name="Cost" unit="$" stroke="#64748b" fontSize={11} tickFormatter={(val) => (val/1000 >= 1 ? (val/1000).toFixed(0) + 'k' : val)} />
                      <YAxis dataKey="accuracy" type="number" name="Accuracy" stroke="#64748b" fontSize={11} domain={['auto', 'auto']} tickFormatter={(val) => (val * 100).toFixed(0) + '%'} />
                      
                      <RechartsTooltip cursor={{ strokeDasharray: '3 3', stroke: '#475569' }} 
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const d = payload[0].payload as MLRun;
                            return (
                              <div className="bg-[#0f172a]/95 backdrop-blur border border-slate-700/80 p-3 rounded-lg shadow-2xl z-50">
                                <p className="text-blue-400 font-bold mb-1 tracking-wider text-xs uppercase">{d.source || 'Run'} <span className="text-slate-500 ml-1">#{d.run_id ? d.run_id.slice(0,6) : ''}</span></p>
                                <p className="text-slate-200 text-xs">Accuracy: <span className="text-emerald-400 font-bold">{(d.accuracy * 100).toFixed(2)}%</span></p>
                                <p className="text-slate-200 text-xs">Cost: <span className="text-blue-400 font-bold">${(d.estimated_cost_usd || 0).toFixed(6)}</span></p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      
                      {/* Plot points conditionally based on source representation */}
                      {source.includes('Both') ? (
                          <>
                             <Scatter name="MLflow runs" data={runs.filter(r => !r.is_pareto_optimal && r.source?.includes('MLflow'))} fill="#64748b" opacity={0.6} shape="circle" />
                             <Scatter name="Optuna runs" data={runs.filter(r => !r.is_pareto_optimal && r.source?.includes('Optuna'))} fill="#10b981" opacity={0.4} shape="circle" />
                          </>
                      ) : (
                          <Scatter name="All Runs" data={runs.filter(r => !r.is_pareto_optimal)} fill="#334155" opacity={0.6} shape="circle" />
                      )}
                      
                      {/* Pareto Line & Points */}
                      <Scatter name="Pareto Front" data={paretoLineData} line={{ stroke: '#3B82F6', strokeWidth: 2.5 }} fill="#2563EB" shape="circle">
                        {
                          paretoLineData.map((entry, index) => {
                            const isOptimal = optimalRun?.run_id === entry.run_id;
                            const color = isOptimal ? '#60A5FA' : (entry.source?.includes('Optuna') && source.includes('Both') ? '#10b981' : '#1D4ED8');
                            return (
                                <Cell key={`cell-${index}`} fill={color} stroke={isOptimal ? '#fff' : 'transparent'} strokeWidth={isOptimal ? 2 : 0} r={isOptimal ? 6 : 4} />
                            );
                          })
                        }
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </section>

                {/* Budget Slider */}
                <section className="h-32 bg-[var(--color-glass-bg)] backdrop-blur rounded-xl border border-[var(--color-glass-border)] p-6 shadow-[0_4px_30px_rgba(0,0,0,0.3)] flex flex-col justify-center">
                  <div className="flex justify-between items-end mb-2 relative z-10">
                    <div>
                       <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase">Maximum Budget Constraint</h3>
                    </div>
                    <div className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded">
                        <span className="text-[16px] text-blue-400 font-bold tracking-wider">${(budget).toLocaleString()}</span>
                    </div>
                  </div>
                  
                  <div className="relative pt-2 pb-1">
                    <input 
                        type="range" 
                        min="0" 
                        max={Math.max(...runs.map(r => r.estimated_cost_usd || 0), 50)*1.1} 
                        step={Math.max(...runs.map(r => r.estimated_cost_usd || 0)) > 1000 ? "10" : "0.5"} 
                        value={budget} 
                        onChange={(e) => setBudget(Number(e.target.value))}
                        className="w-full h-1 bg-slate-700/50 rounded-lg appearance-none cursor-pointer accent-blue-500" 
                    />
                  </div>
                </section>

              </div>

              {/* Right Column (KPIs) - Col 4 */}
              <div className="col-span-1 flex flex-col gap-6 h-full">
                
                <section className="flex-1 bg-[var(--color-glass-bg)] backdrop-blur rounded-xl border border-[var(--color-glass-border)] p-6 shadow-[0_4px_30px_rgba(0,0,0,0.3)] flex flex-col relative overflow-hidden group">
                  <div className="absolute top-0 right-0 w-48 h-48 bg-blue-500/5 rounded-bl-full blur-3xl transition-all duration-700 group-hover:bg-blue-500/10"></div>
                  
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase">Optimal Parameters</h3>
                    <span className="text-[10px] uppercase text-emerald-500 tracking-widest font-bold">Recommended</span>
                  </div>
                  
                  {optimalRun ? (
                    <>
                      <div className="grid grid-cols-2 gap-4 mb-6 z-10">
                        <div>
                          <p className="text-slate-400 text-[10px] xl:text-xs uppercase tracking-wider mb-1">Accuracy</p>
                          <p className="text-white font-[700] text-xl xl:text-2xl tracking-tight">{(optimalRun.accuracy * 100).toFixed(2)}<span className="text-sm text-slate-500">%</span></p>
                        </div>
                        <div>
                          <p className="text-slate-400 text-[10px] xl:text-xs uppercase tracking-wider mb-1">Cost</p>
                          <p className="text-blue-400 font-[700] text-xl xl:text-2xl tracking-tight"><span className="text-sm text-blue-600/70">$</span>{(optimalRun.estimated_cost_usd || 0).toFixed(6)}</p>
                        </div>
                      </div>

                      <div className="border-t border-slate-700/50 pt-5 mt-auto z-10">
                        <div className="space-y-3 font-mono text-[11px] xl:text-xs">
                          <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                            <span className="text-slate-400">Source</span>
                            <span className="text-blue-300 font-bold">{optimalRun.source}</span>
                          </div>
                          <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                            <span className="text-slate-400">Estimators</span>
                            <span className="text-white font-medium">{optimalRun.n_estimators}</span>
                          </div>
                          <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                            <span className="text-slate-400">Max Depth</span>
                            <span className="text-white font-medium">{optimalRun.max_depth || 'None'}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400">Min Split</span>
                            <span className="text-white font-medium">{optimalRun.min_samples_split || '2'}</span>
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-slate-500 text-center text-sm z-10 flex flex-col items-center justify-center h-full">
                        <Activity size={32} className="text-slate-700 mb-2" />
                        No viabile run under limit.
                    </div>
                  )}
                </section>
              </div>
            </div>

            {/* Bottom Section: Optuna Convergence + Data Table */}
            <div className={`grid grid-cols-1 ${source.includes('Optuna') ? 'xl:grid-cols-2' : ''} gap-6`}>
              
              {/* Convergence Chart (Only show if Optuna is active) */}
              {source.includes('Optuna') && (
                <section className="h-[300px] bg-[var(--color-glass-bg)] backdrop-blur rounded-xl border border-[var(--color-glass-border)] p-5 shadow-[0_4px_30px_rgba(0,0,0,0.3)] flex flex-col">
                   <div className="flex justify-between items-center mb-4">
                      <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase">TPE Convergence</h3>
                      <span className="text-[10px] text-slate-400">How the Optuna sampler learned</span>
                   </div>
                   <ResponsiveContainer width="100%" height="85%">
                      <ScatterChart margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                         <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                         <XAxis dataKey="trial_number" type="number" name="Trial" stroke="#64748b" fontSize={11} domain={['dataMin', 'dataMax']} />
                         <YAxis yAxisId="left" dataKey="accuracy" type="number" stroke="#64748b" fontSize={11} domain={['auto', 'auto']} tickFormatter={(v) => (v*100).toFixed(0)+'%'} />
                         <RechartsTooltip cursor={{strokeDasharray: '3 3'}} content={({active, payload}) => {
                             if(active && payload && payload.length) {
                                 const d = payload[0].payload;
                                 return (
                                     <div className="bg-[#0f172a] border border-slate-700 p-2 rounded shadow-xl text-xs">
                                         <p className="font-bold text-white mb-1">Trial #{d.trial_number}</p>
                                         <p className="text-slate-300">Acc: {(d.accuracy*100).toFixed(2)}%</p>
                                         <p className="text-emerald-400 mt-1">Best so far: {(d.best_so_far*100).toFixed(2)}%</p>
                                     </div>
                                 );
                             } return null;
                         }} />
                         {/* We plot a unified line chart using scatter because recharts scatter + line combo requires composed, but scatter with line prop works perfectly. */}
                         <Scatter data={convergenceData} yAxisId="left" fill="#34d399" opacity={0.6} shape="circle" />
                         <Scatter data={convergenceData} yAxisId="left" line={{ stroke: '#10b981', strokeWidth: 2 }} lineJointType="monotoneX" shape={<></>} dataKey="best_so_far" fill="none" />
                      </ScatterChart>
                   </ResponsiveContainer>
                </section>
              )}

              {/* Data Table */}
              <section className="bg-[var(--color-glass-bg)] h-[300px] backdrop-blur rounded-xl border border-[var(--color-glass-border)] flex flex-col overflow-hidden">
                <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-[#0B111D]/50">
                    <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase flex items-center gap-2">
                        <Database size={16} className="text-slate-400"/> Experiment Logs
                    </h3>
                    <button 
                       onClick={handleDownloadCsv}
                       className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded border border-slate-600 transition flex items-center gap-2">
                       <Download size={14}/> CSV Export
                    </button>
                </div>
                <div className="flex-1 overflow-auto custom-scrollbar p-2">
                   <table className="w-full text-left text-xs whitespace-nowrap text-slate-300">
                      <thead className="sticky top-0 bg-[#0f172a] text-slate-400 shadow-sm z-10">
                         <tr>
                            <th className="font-medium p-2 rounded-l">Run ID / Source</th>
                            <th className="font-medium p-2 text-right">Accuracy</th>
                            <th className="font-medium p-2 text-right">Est. Cost</th>
                            <th className="font-medium p-2 text-center">Opt/DOM</th>
                            <th className="font-medium p-2 rounded-r">Params (est, depth, split)</th>
                         </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                          {runs.map((r, i) => (
                              <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                                 <td className="p-2 flex flex-col">
                                     <span className="font-mono text-blue-300">{r.run_id ? r.run_id.slice(0, 10)+ '...' : '—'}</span>
                                     <span className="text-[9px] text-slate-500 uppercase">{r.source} {r.trial_number ? `#${r.trial_number}` : ''}</span>
                                 </td>
                                 <td className="p-2 text-right font-medium text-emerald-400">{(r.accuracy * 100).toFixed(2)}%</td>
                                 <td className="p-2 text-right text-slate-300">${(r.estimated_cost_usd || 0).toFixed(6)}</td>
                                 <td className="p-2 text-center">
                                     {r.is_pareto_optimal ? 
                                        <span className="text-blue-400 font-bold text-[10px] px-1.5 py-0.5 rounded border border-blue-500/20 bg-blue-500/10">Pareto</span> : 
                                        <span className="text-slate-500 text-[10px]">Dominated</span>}
                                 </td>
                                 <td className="p-2 font-mono text-slate-400">
                                     [{r.n_estimators || '-'}, {r.max_depth || '-'}, {r.min_samples_split || '-'}]
                                 </td>
                              </tr>
                          ))}
                      </tbody>
                   </table>
                </div>
              </section>

            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
