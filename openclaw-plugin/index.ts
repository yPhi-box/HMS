/**
 * HMS - Hybrid Memory Server - OpenClaw Plugin
 * 
 * Custom memory system with zero API costs using local embeddings and SQLite.
 * Provides hybrid semantic + keyword search with temporal decay.
 */

interface PluginConfig {
  serverUrl?: string;
  maxResults?: number;
  minScore?: number;
  semanticWeight?: number;
  keywordWeight?: number;
}

interface SearchResult {
  file_path: string;
  line_start: number;
  line_end: number;
  text: string;
  score: number;
  semantic_score: number;
  keyword_score: number;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

export default function register(api: any) {
  const logger = api.logger;

  /**
   * Safe logging - ensures message is always a string
   */
  function log(level: 'info' | 'error', message: string) {
    const msg = String(message);
    if (level === 'error') {
      logger.error(msg);
    } else {
      logger.info(msg);
    }
  }

  /**
   * Get plugin config with defaults
   */
  function getConfig(): Required<PluginConfig> {
    const config = api.config.plugins?.entries?.['openclaw-memory']?.config || {};
    return {
      serverUrl: config.serverUrl || 'http://127.0.0.1:8765',
      maxResults: config.maxResults || 10,
      minScore: config.minScore || 0.0,
      semanticWeight: config.semanticWeight || 0.6,
      keywordWeight: config.keywordWeight || 0.4,
    };
  }

  /**
   * Make HTTP request to memory server
   */
  async function request(endpoint: string, body: any): Promise<any> {
    const config = getConfig();
    const url = `${config.serverUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        const errorMsg = `HTTP ${response.status}: ${errorText}`;
        log('error', `[hms-memory] Memory server request failed: ${url} - ${errorMsg}`);
        throw new Error(errorMsg);
      }

      return await response.json();
    } catch (error: unknown) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      log('error', `[hms-memory] Memory server request failed: ${url} - ${errorMsg}`);
      throw new Error(`Memory server unavailable: ${errorMsg}`);
    }
  }

  /**
   * Format search results for OpenClaw
   */
  function formatResults(results: SearchResult[]): string {
    if (results.length === 0) {
      return 'No results found.';
    }

    const lines: string[] = [];
    
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const fileName = result.file_path.split(/[/\\]/).pop() || result.file_path;
      
      lines.push(`## Result ${i + 1}`);
      lines.push(`**File:** ${fileName}`);
      lines.push(`**Lines:** ${result.line_start}-${result.line_end}`);
      lines.push(`**Score:** ${result.score.toFixed(3)} (semantic: ${result.semantic_score.toFixed(3)}, keyword: ${result.keyword_score.toFixed(3)})`);
      lines.push('');
      lines.push('```');
      lines.push(result.text);
      lines.push('```');
      lines.push('');
    }

    return lines.join('\n');
  }

  /**
   * Register memory_search tool
   */
  api.registerTool({
    name: 'memory_search',
    description: 'MANDATORY: Search memory before answering ANY question about machines, people, credentials, configs, IPs, history, or prior decisions. Do NOT guess, do NOT say "I don\'t know," do NOT ask the user — search first. This tool returns results in under 500ms. There is no excuse to skip it. Searches MEMORY.md, memory/*.md, transcripts, and all indexed workspace files using hybrid semantic + keyword + entity matching with cross-encoder reranking.',
    parameters: {
      type: 'object',
      required: ['query'],
      properties: {
        query: {
          type: 'string',
          description: 'Search query text',
        },
        maxResults: {
          type: 'number',
          description: 'Maximum number of results (default: 10)',
        },
        minScore: {
          type: 'number',
          description: 'Minimum similarity score threshold 0-1 (default: 0.0)',
        },
      },
    },
    async execute(_id: string, params: any) {
      const { query, maxResults, minScore } = params;
      const config = getConfig();

      const json = (text: string) => ({
        content: [{ type: 'text' as const, text: String(text) }]
      });

      log('info', `[hms-memory] memory_search called: query="${query}", maxResults=${maxResults || config.maxResults}`);

      try {
        const response: SearchResponse = await request('/search', {
          query: String(query),
          max_results: maxResults || config.maxResults,
          min_score: minScore !== undefined ? minScore : config.minScore,
          semantic_weight: config.semanticWeight,
          keyword_weight: config.keywordWeight,
        });

        log('info', `[hms-memory] memory_search returned ${response.total} results`);

        const resultText = formatResults(response.results);
        return json(resultText);
      } catch (error: unknown) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        log('error', `[hms-memory] memory_search failed: ${errorMsg}`);
        return json(`Memory search failed: ${errorMsg}`);
      }
    },
  });

  /**
   * Register memory_get tool
   */
  api.registerTool({
    name: 'memory_get',
    description: 'Retrieve specific lines from a memory file. Use after memory_search to get full context for a specific section.',
    parameters: {
      type: 'object',
      required: ['path'],
      properties: {
        path: {
          type: 'string',
          description: 'File path (from memory_search results)',
        },
        from: {
          type: 'number',
          description: 'Starting line number (1-indexed)',
        },
        lines: {
          type: 'number',
          description: 'Number of lines to read',
        },
      },
    },
    async execute(_id: string, params: any) {
      const { path, from, lines } = params;

      const json = (text: string) => ({
        content: [{ type: 'text' as const, text: String(text) }]
      });

      log('info', `[hms-memory] memory_get called: path="${path}", from=${from || 1}, lines=${lines || 'all'}`);

      try {
        // Read file directly using Node.js fs
        const fs = await import('fs');
        const content = await fs.promises.readFile(String(path), 'utf-8');
        const allLines = content.split('\n');

        let startLine = 1;
        let numLines = allLines.length;

        if (from !== undefined) {
          startLine = Math.max(1, Math.min(from, allLines.length));
        }

        if (lines !== undefined) {
          numLines = Math.min(lines, allLines.length - startLine + 1);
        }

        const selectedLines = allLines.slice(startLine - 1, startLine - 1 + numLines);
        const text = selectedLines.join('\n');

        log('info', `[hms-memory] memory_get success: ${selectedLines.length} lines returned (${startLine}-${startLine + selectedLines.length - 1})`);

        return json(text);
      } catch (error: unknown) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        log('error', `[hms-memory] memory_get failed: ${path} - ${errorMsg}`);
        return json(`Failed to read file: ${errorMsg}`);
      }
    },
  });

  log('info', 'HMS - Hybrid Memory Server plugin loaded');
}
