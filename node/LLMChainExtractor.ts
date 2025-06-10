import { FaissStore } from "@langchain/community/vectorstores/faiss";
// import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import "dotenv/config";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
import {ChatOllama, OllamaEmbeddings} from "@langchain/ollama";

process.env.LANGCHAIN_VERBOSE = "true";

async function run() {
  const directory = "../db/kongyiji";
  // const embeddings = new OpenAIEmbeddings();
   const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", // 或 "all-minilm"
  });
  const vectorstore = await FaissStore.load(directory, embeddings);

  // const model = new ChatOpenAI();
  // ✅ 使用本地 Ollama 模型
  const model = new ChatOllama({
    baseUrl: "http://localhost:11434", // Ollama 本地服务地址
    model: "llama3:latest", // 只写模型名
  });
  const compressor = LLMChainExtractor.fromLLM(model);

  const retriever = new ContextualCompressionRetriever({
    baseCompressor: compressor,
    baseRetriever: vectorstore.asRetriever(2),
  });
  const res = await retriever.invoke("茴香豆是做什么用的");
  console.log(res);
}

run();
