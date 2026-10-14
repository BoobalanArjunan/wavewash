import { useState, useEffect, useMemo } from 'react'
import { AudioWaveform, Play, SlidersHorizontal, Sparkles } from 'lucide-react'
import FolderBrowser from './components/FolderBrowser'
import PurposeSelector from './components/PurposeSelector'
import AdvancedSettings from './components/AdvancedSettings'
import QualityPresets from './components/QualityPresets'
import PathsPanel, { PATH_FIELDS, insideOrSame } from './components/PathsPanel'
import RunOptions from './components/RunOptions'
import RunProgress from './components/RunProgress'
import ResultsDashboard from './components/ResultsDashboard'
import { getPurposes, wavCount, startRun, getJobResults, errorMessage } from './api'
import { buildPresetOverrides } from './presets'

const STAGES = { SETUP: 'setup', RUNNING: 'running', RESULTS: 'results' }

const suggestOutput = (input) => {
  return input.replace(/[\\/]+$/, '') + '_QC_output'
}

export default function App() {
  const [stage, setStage] = useState(STAGES.SETUP)
  const [purposes, setPurposes] = useState([])
  const [selectedPurpose, setSelectedPurpose] = useState(null)

  // manual paths (typed/pasted) or browsed -- all four live in the right-hand panel
  const [paths, setPaths] = useState({ input: '', output: '', transcripts: '', metadata: '' })
  const [outputTouched, setOutputTouched] = useState(false)
  const [activeField, setActiveField] = useState('input')
  const [inputInfo, setInputInfo] = useState({ count: null, error: null })
  const [expectedLanguage, setExpectedLanguage] = useState('')

  const [dedupMode, setDedupMode] = useState('near')
  const [denoiseChoice, setDenoiseChoice] = useState(null) // null = follow the purpose's default
  const [extraFormats, setExtraFormats] = useState(false)

  const [settingsMode, setSettingsMode] = useState('simple')
  const [strictness, setStrictness] = useState('balanced')
  const [loudnessTarget, setLoudnessTarget] = useState('default')
  const [advancedOverrides, setAdvancedOverrides] = useState({})

  const [jobId, setJobId] = useState(null)
  const [results, setResults] = useState(null)
  const [runError, setRunError] = useState(null)

  useEffect(() => {
    getPurposes().then(data => {
      setPurposes(data)
      if (data.length) setSelectedPurpose(data[0].key)
    }).catch(e => setRunError(errorMessage(e, 'Could not reach the local backend. Is it still running?')))
  }, [])

  const setPath = (key, value) => {
    setPaths(prev => {
      const next = { ...prev, [key]: value }
      if (key === 'input' && !outputTouched) next.output = value.trim() ? suggestOutput(value.trim()) : ''
      return next
    })
    if (key === 'output') setOutputTouched(true)
  }

  // validate the input path (debounced) whether it was typed or browsed
  useEffect(() => {
    const p = paths.input.trim()
    if (!p) { setInputInfo({ count: null, error: null }); return }
    const t = setTimeout(() => {
      wavCount(p, extraFormats)
        .then(d => setInputInfo({ count: d.count, error: null }))
        .catch(e => setInputInfo({ count: null, error: errorMessage(e, 'Could not read this path.') }))
    }, 350)
    return () => clearTimeout(t)
  }, [paths.input, extraFormats])

  const currentProfile = purposes.find(p => p.key === selectedPurpose)
  const denoise = denoiseChoice ?? currentProfile?.enable_denoise ?? false

  const effectiveOverrides = useMemo(() => {
    const base = settingsMode === 'simple'
      ? buildPresetOverrides(currentProfile, strictness, loudnessTarget)
      : advancedOverrides
    return { ...base, enable_denoise: denoise }
  }, [settingsMode, currentProfile, strictness, loudnessTarget, advancedOverrides, denoise])

  const switchToAdvanced = () => {
    setAdvancedOverrides(buildPresetOverrides(currentProfile, strictness, loudnessTarget))
    setSettingsMode('advanced')
  }

  const chooseDir = (path) => setPath(activeField, path)
  const activeDef = PATH_FIELDS.find(f => f.key === activeField)

  const outputInvalid = paths.output.trim() && paths.input.trim() && insideOrSame(paths.output.trim(), paths.input.trim())
  const blocker =
    !paths.input.trim() ? 'Choose an input folder' :
    inputInfo.error ? inputInfo.error :
    inputInfo.count === 0 ? 'No audio files found in the input' :
    inputInfo.count === null ? 'Checking the input folder…' :
    !paths.output.trim() ? 'Choose an output folder' :
    outputInvalid ? 'Output can’t be inside the input folder' : null
  const canRun = !blocker && !!selectedPurpose

  const handleRun = async () => {
    setRunError(null)
    try {
      const { job_id } = await startRun({
        input_path: paths.input.trim(),
        output_path: paths.output.trim(),
        purpose: selectedPurpose,
        include_extra_formats: extraFormats,
        dedup_mode: dedupMode,
        transcripts_path: paths.transcripts.trim() || undefined,
        metadata_path: paths.metadata.trim() || undefined,
        expected_language: expectedLanguage.trim() || undefined,
        make_split: !!paths.metadata.trim(),
        threshold_overrides: effectiveOverrides,
      })
      setJobId(job_id)
      setStage(STAGES.RUNNING)
    } catch (e) {
      setRunError(errorMessage(e, 'Could not start the run.'))
    }
  }

  const handleJobDone = async (jobStatus) => {
    if (jobStatus.status === 'done') {
      try {
        setResults(await getJobResults(jobId))
      } catch (e) {
        setResults({ status_counts: {}, files: [], error: true, log_tail: errorMessage(e) })
      }
    } else {
      setResults({ status_counts: {}, files: [], error: true, log_tail: jobStatus.log_tail })
    }
    setStage(STAGES.RESULTS)
  }

  const handleReset = () => { setStage(STAGES.SETUP); setResults(null); setJobId(null) }

  return (
    <div className="h-screen flex flex-col bg-[var(--color-bg)] text-[var(--color-text)]">
      <header className="flex items-center gap-3 px-5 py-3.5 border-b border-[var(--color-border)] shrink-0">
        <div className="p-1.5 rounded-md bg-[var(--color-accent)]"><AudioWaveform size={16} className="text-[#14161A]" /></div>
        <h1 className="text-sm font-bold tracking-tight">WaveWash</h1>
      </header>

      {stage === STAGES.SETUP && (
        <div className="flex-1 flex overflow-hidden">
          <div className="w-96 border-r border-[var(--color-border)] flex flex-col overflow-y-auto shrink-0">
            <div className="p-4 border-b border-[var(--color-border)]">
              <label className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wide">Purpose</label>
              <div className="mt-2"><PurposeSelector purposes={purposes} selected={selectedPurpose}
                onSelect={(k) => { setSelectedPurpose(k); setDenoiseChoice(null) }} /></div>
            </div>

            <div className="p-4 border-b border-[var(--color-border)]">
              <RunOptions dedupMode={dedupMode} onDedupChange={setDedupMode}
                denoise={denoise} onDenoiseChange={setDenoiseChoice}
                extraFormats={extraFormats} onExtraFormatsChange={setExtraFormats} />
            </div>

            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wide">Quality settings</label>
                <div className="flex rounded-md overflow-hidden border border-[var(--color-border)]">
                  <button onClick={() => setSettingsMode('simple')}
                    className={`flex items-center gap-1 px-2 py-1 text-[11px] font-medium transition-colors ${settingsMode === 'simple' ? 'bg-[var(--color-accent)] text-[#14161A]' : 'text-[var(--color-text-muted)] hover:bg-[var(--color-panel-raised)]'}`}>
                    <Sparkles size={11} /> Simple
                  </button>
                  <button onClick={switchToAdvanced}
                    className={`flex items-center gap-1 px-2 py-1 text-[11px] font-medium transition-colors ${settingsMode === 'advanced' ? 'bg-[var(--color-accent)] text-[#14161A]' : 'text-[var(--color-text-muted)] hover:bg-[var(--color-panel-raised)]'}`}>
                    <SlidersHorizontal size={11} /> Advanced
                  </button>
                </div>
              </div>
              {settingsMode === 'simple'
                ? <QualityPresets strictness={strictness} onStrictnessChange={setStrictness} loudness={loudnessTarget} onLoudnessChange={setLoudnessTarget} />
                : <AdvancedSettings profile={currentProfile} overrides={advancedOverrides} onChange={setAdvancedOverrides} />}
            </div>

            <div className="mt-auto p-4 border-t border-[var(--color-border)]">
              {runError && <div role="alert" className="mb-2 text-xs text-[var(--color-fail)] leading-snug">{runError}</div>}
              <button onClick={handleRun} disabled={!canRun}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-lg bg-[var(--color-accent)] text-[#14161A] font-semibold text-sm hover:brightness-110 transition-all disabled:opacity-40 disabled:cursor-not-allowed">
                <Play size={15} fill="currentColor" /> Run quality check
              </button>
              {blocker && <div className="mt-2 text-[11px] text-center text-[var(--color-text-faint)]">{blocker}</div>}
            </div>
          </div>

          <div className="flex-1 flex flex-col overflow-hidden min-w-0">
            <PathsPanel paths={paths} onChange={setPath} activeField={activeField} onActivate={setActiveField}
              inputInfo={inputInfo} expectedLanguage={expectedLanguage} onLanguageChange={setExpectedLanguage} />
            <div className="flex-1 min-h-0">
              <FolderBrowser key={`${activeDef.mode}:${(activeDef.fileExts || []).join()}`} mode={activeDef.mode} fileExts={activeDef.fileExts}
                targetLabel={activeDef.label.toLowerCase()} onPick={chooseDir} />
            </div>
          </div>
        </div>
      )}

      {stage === STAGES.RUNNING && (
        <div className="flex-1 overflow-hidden"><RunProgress jobId={jobId} totalFiles={inputInfo.count} onDone={handleJobDone} /></div>
      )}
      {stage === STAGES.RESULTS && results && (
        <div className="flex-1 overflow-hidden"><ResultsDashboard results={results} jobId={jobId} onReset={handleReset} /></div>
      )}
    </div>
  )
}
