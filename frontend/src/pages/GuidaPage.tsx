import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BookOpen,
  Zap,
  Cpu,
  MessageSquare,
  HelpCircle,
  BookMarked,
  Settings,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  FolderOpen,
  Bot,
  Brain,
  FileCheck,
  Users,
} from "lucide-react";

type TabId =
  | "quickstart"
  | "features"
  | "models"
  | "agent_examples"
  | "faq"
  | "glossary"
  | "requirements"
  | "privacy";

const TABS: { id: TabId; icon: React.ReactNode }[] = [
  { id: "quickstart", icon: <Zap className="h-4 w-4" /> },
  { id: "features", icon: <BookOpen className="h-4 w-4" /> },
  { id: "models", icon: <Cpu className="h-4 w-4" /> },
  { id: "agent_examples", icon: <MessageSquare className="h-4 w-4" /> },
  { id: "faq", icon: <HelpCircle className="h-4 w-4" /> },
  { id: "glossary", icon: <BookMarked className="h-4 w-4" /> },
  { id: "requirements", icon: <Settings className="h-4 w-4" /> },
  { id: "privacy", icon: <ShieldCheck className="h-4 w-4" /> },
];

function FaqItem({ question, answer }: { question: string; answer: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-3 text-left text-sm font-medium hover:bg-muted/50 transition-colors"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{question}</span>
        {open ? (
          <ChevronUp className="h-4 w-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
        )}
      </button>
      {open && (
        <div className="px-4 pb-4 pt-1 text-sm text-muted-foreground border-t bg-muted/20">
          {answer}
        </div>
      )}
    </div>
  );
}

function GlossaryItem({ term, def }: { term: string; def: string }) {
  return (
    <div className="flex gap-3 py-2 border-b last:border-0">
      <span className="font-mono font-semibold text-sm text-primary w-28 shrink-0">
        {term}
      </span>
      <span className="text-sm text-muted-foreground">{def}</span>
    </div>
  );
}

function ExampleGroup({
  label,
  examples,
  color,
}: {
  label: string;
  examples: string[];
  color: string;
}) {
  return (
    <div className="space-y-2">
      <Badge variant="outline" className={`text-xs font-semibold ${color}`}>
        {label}
      </Badge>
      <ul className="space-y-1">
        {examples.map((ex, i) => (
          <li
            key={i}
            className="text-sm bg-muted/40 rounded px-3 py-1.5 font-mono"
          >
            "{ex}"
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function GuidaPage() {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<TabId>("quickstart");

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">{t("guide.title")}</h1>
        <p className="text-muted-foreground mt-1">{t("guide.subtitle")}</p>
      </div>

      {/* Tab bar */}
      <div className="flex flex-wrap gap-1 border-b pb-0">
        {TABS.map(({ id, icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium rounded-t transition-colors border-b-2 -mb-px ${
              activeTab === id
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {icon}
            {t(`guide.tabs.${id}`)}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="space-y-4">
        {/* ── QUICK START ── */}
        {activeTab === "quickstart" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">
              {t("guide.quickstart.title")}
            </h2>
            <div className="grid sm:grid-cols-2 gap-4">
              {(
                [
                  { key: "step1", icon: <FolderOpen className="h-5 w-5 text-blue-500" /> },
                  { key: "step2", icon: <BookOpen className="h-5 w-5 text-green-500" /> },
                  { key: "step3", icon: <Bot className="h-5 w-5 text-purple-500" /> },
                  { key: "step4", icon: <FileCheck className="h-5 w-5 text-orange-500" /> },
                ] as const
              ).map(({ key, icon }) => (
                <Card key={key}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold flex items-center gap-2">
                      {icon}
                      {t(`guide.quickstart.${key}_title`)}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      {t(`guide.quickstart.${key}_desc`)}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* ── FEATURES ── */}
        {activeTab === "features" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">
              {t("guide.features.title")}
            </h2>
            <div className="space-y-3">
              {(
                [
                  { key: "projects", icon: <FolderOpen className="h-5 w-5 text-blue-500" /> },
                  { key: "agent", icon: <Bot className="h-5 w-5 text-purple-500" /> },
                  { key: "rag", icon: <Brain className="h-5 w-5 text-green-500" /> },
                  { key: "compliance", icon: <FileCheck className="h-5 w-5 text-orange-500" /> },
                  { key: "admin", icon: <Users className="h-5 w-5 text-gray-500" /> },
                ] as const
              ).map(({ key, icon }) => (
                <Card key={key}>
                  <CardContent className="pt-4 flex gap-3">
                    <div className="mt-0.5 shrink-0">{icon}</div>
                    <div>
                      <p className="font-semibold text-sm">
                        {t(`guide.features.${key}_title`)}
                      </p>
                      <p className="text-sm text-muted-foreground mt-0.5">
                        {t(`guide.features.${key}_desc`)}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* ── MODELS ── */}
        {activeTab === "models" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">
              {t("guide.models.title")}
            </h2>
            <p className="text-sm text-muted-foreground">
              {t("guide.models.intro")}
            </p>
            <div className="space-y-3">
              {(["llm", "vlm", "ocr", "embed", "openai"] as const).map(
                (key) => (
                  <Card key={key}>
                    <CardContent className="pt-4">
                      <p className="font-semibold text-sm">
                        {t(`guide.models.${key}_label`)}
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {t(`guide.models.${key}_desc`)}
                      </p>
                    </CardContent>
                  </Card>
                )
              )}
            </div>
            <Card className="border-dashed">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">
                  {t("guide.models.fallback_title")}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
                  {(["fallback_1", "fallback_2", "fallback_3"] as const).map(
                    (k) => (
                      <li key={k}>{t(`guide.models.${k}`)}</li>
                    )
                  )}
                </ul>
              </CardContent>
            </Card>
          </div>
        )}

        {/* ── AGENT EXAMPLES ── */}
        {activeTab === "agent_examples" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">
              {t("guide.agent_examples.title")}
            </h2>
            <p className="text-sm text-muted-foreground">
              {t("guide.agent_examples.intro")}
            </p>
            <div className="grid sm:grid-cols-2 gap-4">
              <ExampleGroup
                label={t("guide.agent_examples.cat_transcription")}
                color="text-blue-600 border-blue-300"
                examples={[
                  t("guide.agent_examples.transcription_1"),
                  t("guide.agent_examples.transcription_2"),
                ]}
              />
              <ExampleGroup
                label={t("guide.agent_examples.cat_signature")}
                color="text-purple-600 border-purple-300"
                examples={[
                  t("guide.agent_examples.signature_1"),
                  t("guide.agent_examples.signature_2"),
                  t("guide.agent_examples.signature_3"),
                ]}
              />
              <ExampleGroup
                label={t("guide.agent_examples.cat_layout")}
                color="text-green-600 border-green-300"
                examples={[
                  t("guide.agent_examples.layout_1"),
                  t("guide.agent_examples.layout_2"),
                  t("guide.agent_examples.layout_3"),
                  t("guide.agent_examples.layout_4"),
                ]}
              />
              <ExampleGroup
                label={t("guide.agent_examples.cat_graphology")}
                color="text-orange-600 border-orange-300"
                examples={[
                  t("guide.agent_examples.graphology_1"),
                  t("guide.agent_examples.graphology_2"),
                ]}
              />
              <ExampleGroup
                label={t("guide.agent_examples.cat_dating")}
                color="text-yellow-700 border-yellow-300"
                examples={[t("guide.agent_examples.dating_1")]}
              />
              <ExampleGroup
                label={t("guide.agent_examples.cat_kb")}
                color="text-teal-600 border-teal-300"
                examples={[
                  t("guide.agent_examples.kb_1"),
                  t("guide.agent_examples.kb_2"),
                ]}
              />
              <ExampleGroup
                label={t("guide.agent_examples.cat_compliance")}
                color="text-red-600 border-red-300"
                examples={[t("guide.agent_examples.compliance_1")]}
              />
            </div>
          </div>
        )}

        {/* ── FAQ ── */}
        {activeTab === "faq" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">{t("guide.faq.title")}</h2>
            <div className="space-y-2">
              {(["1", "2", "3", "4", "5", "6", "7"] as const).map((n) => (
                <FaqItem
                  key={n}
                  question={t(`guide.faq.q${n}`)}
                  answer={t(`guide.faq.a${n}`)}
                />
              ))}
            </div>
          </div>
        )}

        {/* ── GLOSSARY ── */}
        {activeTab === "glossary" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">
              {t("guide.glossary.title")}
            </h2>
            <Card>
              <CardContent className="pt-4">
                {(
                  [
                    "llm",
                    "vlm",
                    "ocr",
                    "htr",
                    "rag",
                    "embedding",
                    "hog",
                    "signet",
                    "enfsi",
                    "bpm",
                    "perizia",
                  ] as const
                ).map((key) => (
                  <GlossaryItem
                    key={key}
                    term={t(`guide.glossary.${key}_term`)}
                    def={t(`guide.glossary.${key}_def`)}
                  />
                ))}
              </CardContent>
            </Card>
          </div>
        )}

        {/* ── REQUIREMENTS ── */}
        {activeTab === "requirements" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">
              {t("guide.requirements.title")}
            </h2>
            <div className="space-y-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    {t("guide.requirements.ollama_title")}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {t("guide.requirements.ollama_desc")}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    {t("guide.requirements.models_title")}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-1 text-sm font-mono">
                    {(
                      ["model_llm", "model_embed", "model_vlm"] as const
                    ).map((k) => (
                      <li
                        key={k}
                        className="bg-muted/40 rounded px-3 py-1.5"
                      >
                        {t(`guide.requirements.${k}`)}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    {t("guide.requirements.hw_title")}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
                    <li>{t("guide.requirements.hw_ram")}</li>
                    <li>{t("guide.requirements.hw_disk")}</li>
                  </ul>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    {t("guide.requirements.openai_title")}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {t("guide.requirements.openai_desc")}
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* ── PRIVACY ── */}
        {activeTab === "privacy" && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">
              {t("guide.privacy.title")}
            </h2>
            <div className="space-y-3">
              {(
                [
                  { key: "local", icon: <ShieldCheck className="h-5 w-5 text-green-500" /> },
                  { key: "openai", icon: <Cpu className="h-5 w-5 text-blue-500" /> },
                  { key: "storage", icon: <Settings className="h-5 w-5 text-gray-500" /> },
                  { key: "audit", icon: <BookOpen className="h-5 w-5 text-orange-500" /> },
                ] as const
              ).map(({ key, icon }) => (
                <Card key={key}>
                  <CardContent className="pt-4 flex gap-3">
                    <div className="mt-0.5 shrink-0">{icon}</div>
                    <div>
                      <p className="font-semibold text-sm">
                        {t(`guide.privacy.${key}_title`)}
                      </p>
                      <p className="text-sm text-muted-foreground mt-0.5">
                        {t(`guide.privacy.${key}_desc`)}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
