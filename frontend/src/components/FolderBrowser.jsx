import { useState, useEffect, useCallback } from 'react'
import { Folder, FolderOpen, ChevronRight, Home, ArrowUp, AudioWaveform, FileText, CornerDownLeft } from 'lucide-react'
import { browse, errorMessage } from '../api'

/**
 * mode 'dir'  : navigate folders; the bottom button picks the current folder.
 * mode 'file' : folders + files matching `fileExts`; clicking a file picks it.
 * The path bar is editable -- paste or type any path and press Enter to jump there.
 */
export default function FolderBrowser({ mode = 'dir', targetLabel, fileExts = [], startPath, onPick }) {
  const [currentPath, setCurrentPath] = useState(null)
  const [pathText, setPathText] = useState('')
  const [entries, setEntries] = useState([])
  const [parentPath, setParentPath] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = useCallback(async (path) => {
    setLoading(true)
    setError(null)
    try {
      const data = await browse(path)
      setCurrentPath(data.current_path)
      setPathText(data.current_path)
      setParentPath(data.parent_path)
      setEntries(data.entries)
    } catch (e) {
      setError(errorMessage(e, 'Could not read this folder.'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load(startPath || null) }, [])

  const visible = entries.filter(e => e.is_dir || (mode === 'file' && fileExts.includes(e.ext)))

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-[var(--color-border)]">
        <button onClick={() => load(null)} title="Home"
          className="p-1.5 rounded hover:bg-[var(--color-panel-raised)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors">
          <Home size={15} />
        </button>
        <button onClick={() => parentPath && load(parentPath)} disabled={!parentPath} title="Up one level"
          className="p-1.5 rounded hover:bg-[var(--color-panel-raised)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors disabled:opacity-30 disabled:hover:bg-transparent">
          <ArrowUp size={15} />
        </button>
        <input
          value={pathText}
          onChange={(e) => setPathText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && load(pathText)}
          spellCheck={false}
          aria-label="Browser location"
          placeholder="Type or paste a folder path, then press Enter"
          className="flex-1 min-w-0 bg-[var(--color-panel)] border border-[var(--color-border)] rounded-md px-2.5 py-1.5 text-xs font-mono-data text-[var(--color-text)] placeholder:text-[var(--color-text-faint)] focus:outline-none focus:border-[var(--color-accent)]"
        />
        <button onClick={() => load(pathText)} title="Go to this path"
          className="p-1.5 rounded hover:bg-[var(--color-panel-raised)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors">
          <CornerDownLeft size={15} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto py-1 min-h-0">
        {loading && <div className="px-4 py-6 text-sm text-[var(--color-text-faint)]">Reading folder…</div>}
        {error && <div className="px-4 py-6 text-sm text-[var(--color-fail)]">{error}</div>}
        {!loading && !error && visible.length === 0 && (
          <div className="px-4 py-6 text-sm text-[var(--color-text-faint)]">
            {mode === 'file' ? 'No folders or matching files here.' : 'No subfolders here.'}
          </div>
        )}
        {!loading && visible.map((entry) => entry.is_dir ? (
          <button key={entry.path} onClick={() => load(entry.path)}
            className="w-full flex items-center gap-2.5 px-4 py-2 text-left hover:bg-[var(--color-panel-raised)] transition-colors group">
            <Folder size={16} className="text-[var(--color-text-faint)] group-hover:text-[var(--color-accent)] shrink-0 transition-colors" />
            <span className="text-sm flex-1 truncate">{entry.name}</span>
            {entry.wav_count > 0 && (
              <span className="flex items-center gap-1 text-xs font-mono-data text-[var(--color-text-faint)] shrink-0" title="audio files directly in this folder">
                <AudioWaveform size={11} />{entry.wav_count}
              </span>
            )}
            <ChevronRight size={14} className="text-[var(--color-text-faint)] opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
          </button>
        ) : (
          <button key={entry.path} onClick={() => onPick(entry.path)}
            className="w-full flex items-center gap-2.5 px-4 py-2 text-left hover:bg-[var(--color-accent)]/10 transition-colors group">
            <FileText size={16} className="text-[var(--color-accent)] shrink-0" />
            <span className="text-sm flex-1 truncate font-mono-data">{entry.name}</span>
            <span className="text-[10px] text-[var(--color-accent)] opacity-0 group-hover:opacity-100">select</span>
          </button>
        ))}
      </div>

      <div className="p-3 border-t border-[var(--color-border)]">
        {mode === 'dir' ? (
          <button onClick={() => currentPath && onPick(currentPath)} disabled={!currentPath}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-[var(--color-accent)] text-[#14161A] font-semibold text-sm hover:brightness-110 transition-all disabled:opacity-40">
            <FolderOpen size={15} />
            Use this folder as {targetLabel}
          </button>
        ) : (
          <div className="text-xs text-center text-[var(--color-text-muted)] py-1.5">
            Click a file above to use it as {targetLabel}
          </div>
        )}
      </div>
    </div>
  )
}
