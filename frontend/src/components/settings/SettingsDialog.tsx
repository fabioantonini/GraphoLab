import { useTranslation } from "react-i18next"
import { RefreshCw, Check, Trash2 } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

interface SettingsDialogProps {
  open: boolean
  onClose: () => void
  // Language
  currentLang: string
  onToggleLang: () => void
  // Ollama + OpenAI models
  models: string[]
  ollamaUp: boolean | null
  refreshing: boolean
  onRefresh: () => void
  openaiModels: { llm: string[]; vlm: string[]; embed: string[] }
  // LLM model
  currentModel: string
  onModelChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
  // OCR model
  currentOcrModel: string
  onOcrModelChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
  // VLM model
  currentVlmModel: string
  onVlmModelChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
  // Embedding model
  openaiKeyConfigured: boolean
  currentEmbedModel: string
  onEmbedModelChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
  // OpenAI key
  showKeyInput: boolean
  setShowKeyInput: (v: boolean) => void
  keyDraft: string
  setKeyDraft: (v: string) => void
  savingKey: boolean
  keyError: string
  onSaveKey: (e: React.FormEvent) => void
  onDeleteKey: () => void
}

const OCR_MODELS = ["easyocr", "vlm", "paddleocr", "trocr"]

export default function SettingsDialog({
  open,
  onClose,
  models,
  ollamaUp,
  refreshing,
  onRefresh,
  openaiModels,
  currentModel,
  onModelChange,
  currentOcrModel,
  onOcrModelChange,
  currentVlmModel,
  onVlmModelChange,
  openaiKeyConfigured,
  currentEmbedModel,
  onEmbedModelChange,
  showKeyInput,
  setShowKeyInput,
  keyDraft,
  setKeyDraft,
  savingKey,
  keyError,
  onSaveKey,
  onDeleteKey,
  currentLang,
  onToggleLang,
}: SettingsDialogProps) {
  const { t } = useTranslation()

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onClose() }}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <div className="flex items-center justify-between pr-6">
            <DialogTitle>{t("nav.settings")}</DialogTitle>
            <button
              onClick={onRefresh}
              disabled={refreshing}
              title={t("config.refresh")}
              className="text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} />
            </button>
          </div>
        </DialogHeader>

        <div className="px-6 pb-6 pt-4 space-y-4">

          {/* LLM model */}
          <div>
            <label className="block text-sm font-medium mb-1.5">{t("config.model_label")}</label>
            {ollamaUp === false && openaiModels.llm.length === 0 ? (
              <p className="text-xs text-destructive">{t("config.model_offline")}</p>
            ) : models.length === 0 && openaiModels.llm.length === 0 ? (
              <p className="text-xs text-muted-foreground">{t("config.model_loading")}</p>
            ) : (
              <select
                value={currentModel}
                onChange={onModelChange}
                className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              >
                {models.length > 0 && (
                  <optgroup label="Ollama">
                    {models.map(m => <option key={m} value={m}>{m}</option>)}
                  </optgroup>
                )}
                {openaiModels.llm.length > 0 && (
                  <optgroup label="OpenAI">
                    {openaiModels.llm.map(m => <option key={m} value={m}>{m}</option>)}
                  </optgroup>
                )}
              </select>
            )}
          </div>

          {/* OCR model */}
          <div>
            <label className="block text-sm font-medium mb-1.5">{t("config.ocr_model_label")}</label>
            <select
              value={currentOcrModel}
              onChange={onOcrModelChange}
              className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
            >
              {OCR_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          {/* VLM model */}
          <div>
            <label className="block text-sm font-medium mb-1.5">{t("config.vlm_model_label")}</label>
            {models.length === 0 && openaiModels.vlm.length === 0 ? (
              <p className="text-xs text-muted-foreground">{t("config.model_loading")}</p>
            ) : (
              <select
                value={currentVlmModel}
                onChange={onVlmModelChange}
                className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              >
                {models.length > 0 && (
                  <optgroup label="Ollama">
                    {models.map(m => <option key={m} value={m}>{m}</option>)}
                  </optgroup>
                )}
                {openaiModels.vlm.length > 0 && (
                  <optgroup label="OpenAI">
                    {openaiModels.vlm.map(m => <option key={m} value={m}>{m}</option>)}
                  </optgroup>
                )}
              </select>
            )}
          </div>

          {/* Embedding model (only when OpenAI key configured) */}
          {openaiKeyConfigured && (
            <div>
              <label className="block text-sm font-medium mb-1.5">{t("config.embed_model_label")}</label>
              <select
                value={currentEmbedModel}
                onChange={onEmbedModelChange}
                className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <optgroup label="Ollama">
                  <option value="nomic-embed-text">nomic-embed-text</option>
                </optgroup>
                <optgroup label="OpenAI">
                  {openaiModels.embed.map(m => <option key={m} value={m}>{m}</option>)}
                </optgroup>
              </select>
            </div>
          )}

          {/* OpenAI API key */}
          <div className="pt-2 border-t">
            <label className="block text-sm font-medium mb-1.5">{t("config.openai_key_label")}</label>
            {openaiKeyConfigured && !showKeyInput ? (
              <div className="flex items-center gap-2">
                <span className="text-sm text-green-600 dark:text-green-400">{t("config.openai_key_ok")}</span>
                <button
                  className="text-xs text-muted-foreground underline"
                  onClick={() => setShowKeyInput(true)}
                >
                  {t("config.openai_key_change")}
                </button>
                <button
                  className="text-xs text-destructive hover:opacity-70 transition-opacity ml-auto"
                  onClick={onDeleteKey}
                  title={t("config.openai_key_remove")}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ) : !showKeyInput ? (
              <div>
                <p className="text-xs text-muted-foreground mb-1">{t("config.openai_key_missing")}</p>
                <button
                  className="text-sm text-primary underline"
                  onClick={() => setShowKeyInput(true)}
                >
                  {t("config.openai_key_add")}
                </button>
              </div>
            ) : null}
            {showKeyInput && (
              <form onSubmit={onSaveKey} className="flex gap-1.5 mt-1">
                <input
                  type="password"
                  value={keyDraft}
                  onChange={e => { setKeyDraft(e.target.value) }}
                  placeholder="sk-…"
                  className="flex-1 min-w-0 rounded border bg-background px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                />
                <Button
                  type="submit"
                  size="icon"
                  className="h-7 w-7 shrink-0"
                  disabled={savingKey || !keyDraft.trim()}
                >
                  <Check className="h-3.5 w-3.5" />
                </Button>
              </form>
            )}
            {keyError && <p className="text-xs text-destructive mt-1">{keyError}</p>}
          </div>

          {/* Language */}
          <div className="pt-2 border-t flex items-center justify-between">
            <label className="text-sm font-medium">{t("config.language_label")}</label>
            <button
              onClick={onToggleLang}
              className="text-sm text-primary underline"
            >
              {currentLang === "it" ? "English" : "Italiano"}
            </button>
          </div>

        </div>
      </DialogContent>
    </Dialog>
  )
}
