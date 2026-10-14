import { useState } from 'react'
import { ChevronDown, RotateCcw } from 'lucide-react'

const NUMERIC_FIELDS = [
  { key: 'min_snr_db', label: 'Minimum SNR', unit: 'dB', min: 0, max: 40, step: 0.5,
    help: 'How much clean signal is required relative to background noise. Lower = accepts noisier recordings.' },
  { key: 'max_clipped_ratio', label: 'Max clipped samples', unit: '%', min: 0, max: 2, step: 0.01, scale: 100,
    help: 'How much of a file can be distorted (hitting maximum volume) before it’s rejected.' },
  { key: 'max_duration_sec', label: 'Max clip duration', unit: 's', min: 1, max: 600, step: 1,
    help: 'Longest a single audio file is allowed to be.' },
  { key: 'min_duration_sec', label: 'Min clip duration', unit: 's', min: 0, max: 10, step: 0.1,
    help: 'Shortest a single audio file is allowed to be.' },
  { key: 'max_silence_ratio', label: 'Max silence ratio', unit: '%', min: 0, max: 100, step: 1, scale: 100,
    help: 'How much of a file can be silence before it’s considered empty rather than useful audio.' },
  { key: 'target_lufs', label: 'Target loudness', unit: 'LUFS', min: -36, max: -6, step: 0.5,
    help: 'The volume level files get normalized to. More negative = quieter target.' },
  { key: 'lossy_cliff_db_per_octave', label: 'Compression / filter sensitivity', unit: ' dB/oct', min: 15, max: 120, step: 1,
    help: 'How sharp the high-frequency drop-off must be before a file is flagged as compressed or low-pass filtered. Lower = flags more files. Natural instruments measure about 9–22; filters and codecs 49+.' },
  { key: 'declip_max_run_ms', label: 'Declip repair window', unit: 'ms', min: 1, max: 40, step: 1,
    help: 'The longest stretch of distortion the tool will attempt to repair. Longer stretches are left alone rather than faked.' },
]

const TOGGLE_FIELDS = [
  { key: 'enable_reverb_check', label: 'Check for reverb / room echo' },
  { key: 'enable_rolloff_check', label: 'Check for compression / low-pass filtering' },
  { key: 'enable_crest_factor_check', label: 'Check dynamic range (crest factor)' },
  { key: 'reject_multi_speaker', label: 'Reject suspected multi-speaker audio' },
  { key: 'auto_normalize_loudness', label: 'Auto-normalize loudness' },
  { key: 'attempt_declip', label: 'Attempt to repair clipping' },
]

export default function AdvancedSettings({ profile, overrides, onChange }) {
  const [open, setOpen] = useState(false)
  if (!profile) return null

  const current = (key) => overrides[key] !== undefined ? overrides[key] : profile[key]
  const isOverridden = (key) => overrides[key] !== undefined

  const setValue = (key, value) => onChange({ ...overrides, [key]: value })
  const resetField = (key) => {
    const next = { ...overrides }
    delete next[key]
    onChange(next)
  }

  return (
    <div className="border border-[var(--color-border)] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-[var(--color-panel)] hover:bg-[var(--color-panel-raised)] transition-colors"
      >
        <span className="text-sm font-medium">Advanced settings</span>
        <div className="flex items-center gap-2">
          {Object.keys(overrides).length > 0 && (
            <span className="text-xs font-mono-data text-[var(--color-accent)]">{Object.keys(overrides).length} changed</span>
          )}
          <ChevronDown size={15} className={`text-[var(--color-text-muted)] transition-transform ${open ? 'rotate-180' : ''}`} />
        </div>
      </button>

      {open && (
        <div className="p-3 space-y-4 bg-[var(--color-bg)]">
          <div className="space-y-3">
            {NUMERIC_FIELDS.map((f) => {
              const scale = f.scale || 1
              const rawValue = current(f.key)
              if (rawValue === null || rawValue === undefined) return null
              const displayValue = (rawValue * scale).toFixed(f.step < 1 ? 2 : 0)
              return (
                <div key={f.key}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-[var(--color-text-muted)]">{f.label}</label>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-mono-data text-[var(--color-text)]">{displayValue}{f.unit}</span>
                      {isOverridden(f.key) && (
                        <button onClick={() => resetField(f.key)} title="Reset to default">
                          <RotateCcw size={11} className="text-[var(--color-accent)]" />
                        </button>
                      )}
                    </div>
                  </div>
                  <input
                    type="range"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={rawValue * scale}
                    onChange={(e) => setValue(f.key, parseFloat(e.target.value) / scale)}
                    className="w-full accent-[var(--color-accent)] h-1"
                  />
                  <p className="text-[10px] text-[var(--color-text-faint)] mt-1 leading-snug">{f.help}</p>
                </div>
              )
            })}
          </div>

          <div className="border-t border-[var(--color-border)] pt-3 space-y-2">
            {TOGGLE_FIELDS.map((f) => {
              const value = current(f.key)
              if (value === null || value === undefined) return null
              return (
                <label key={f.key} className="flex items-center justify-between text-xs cursor-pointer py-1">
                  <span className="text-[var(--color-text-muted)]">{f.label}</span>
                  <input
                    type="checkbox"
                    checked={!!value}
                    onChange={(e) => setValue(f.key, e.target.checked)}
                    className="accent-[var(--color-accent)] w-3.5 h-3.5"
                  />
                </label>
              )
            })}
          </div>

          {Object.keys(overrides).length > 0 && (
            <button
              onClick={() => onChange({})}
              className="text-xs text-[var(--color-fail)] hover:underline"
            >
              Reset all to defaults
            </button>
          )}
        </div>
      )}
    </div>
  )
}
