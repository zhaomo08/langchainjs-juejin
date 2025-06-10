import { FaissStore } from "@langchain/community/vectorstores/faiss";
import {ChatOllama} from "@langchain/ollama";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";
import "faiss-node";
import "dotenv/config";
import {OllamaEmbeddings} from "@langchain/ollama";

async function run() {
  const directory = "../db/kongyiji";
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", // 或 "all-minilm"
  });
  // const embeddings = new OpenAIEmbeddings();
  const vectorstore = await FaissStore.load(directory, embeddings);

  // const model = new ChatOpenAI();

   // ✅ 使用本地 Ollama 模型
  const model = new ChatOllama({
    baseUrl: "http://localhost:11434", // Ollama 本地服务地址
    model: "llama3:latest", // 只写模型名
  });
  const retriever = MultiQueryRetriever.fromLLM({
    llm: model,
    retriever: vectorstore.asRetriever(3),
    queryCount: 3,
    verbose: true,
  });
  const res = await retriever.invoke("茴香豆是做什么用的");

  console.log(res);
}

run();
