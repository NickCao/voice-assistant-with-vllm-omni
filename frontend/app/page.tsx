import Image from "next/image";
import VoiceAssistant from "@/components/VoiceAssistant";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-8 p-8">
      <Image
        src="/redhat-logo.svg"
        alt="Red Hat"
        width={160}
        height={48}
        priority
      />
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight">Voice Assistant</h1>
        <p className="mt-2 text-zinc-400">
          Powered by Qwen3-Omni &amp; vLLM-Omni
        </p>
      </div>
      <VoiceAssistant />
    </main>
  );
}
