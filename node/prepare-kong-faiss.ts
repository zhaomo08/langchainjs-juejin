import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
// import { OpenAIEmbeddings } from "@langchain/openai";
import { OllamaEmbeddings } from "@langchain/ollama"; // ✅ 新路径！


const run = async () => {
  const loader = new TextLoader("../data/kong.txt");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", // 或 "all-minilm"
  });
  const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

  const directory = "../db/kongyiji";
  await vectorStore.save(directory);
};

run();
