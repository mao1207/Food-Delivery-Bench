
# Agent Performance Analysis Report

## Executive Summary
This report analyzes the performance of 8 AI agents in a food delivery simulation environment.

## Key Findings

### Performance Comparison Table

| Agent ID | Model | Net Growth ($) | Completed Orders | Timeout Orders | Avg Stars | VLM Total Calls | VLM Success Calls | VLM Success Rate |
|----------|-------|----------------|------------------|----------------|-----------|-----------------|-------------------|------------------|
| 8 | meta-llama/llama-3.2-90b-vision-instruct | 68.56 | 9 | 1 | 4.22 | 59 | 58 | 98.3% |
| 7 | mistralai/mistral-medium-3.1 | 51.72 | 6 | 0 | 4.83 | 62 | 58 | 93.5% |
| 1 | qwen/qwen2.5-vl-32b-instruct | 11.97 | 1 | 0 | 5.00 | 63 | 19 | 30.2% |
| 5 | google/gemini-2.5-flash-lite | 6.04 | 2 | 1 | 4.50 | 57 | 57 | 100.0% |
| 3 | meta-llama/llama-4-maverick | 0.84 | 1 | 1 | 3.00 | 17 | 16 | 94.1% |
| 2 | qwen/qwen-2.5-vl-7b-instruct | 0.00 | 0 | 0 | 0.00 | 67 | 61 | 91.0% |
| 6 | meta-llama/llama-4-scout | 0.00 | 0 | 0 | 0.00 | 83 | 71 | 85.5% |
| 4 | mistralai/mistral-small-3.2-24b-instruct | -0.77 | 2 | 2 | 3.00 | 76 | 76 | 100.0% |

### Additional Performance Metrics

| Agent ID | Model | Active Hours | Orders/Hour | Temp OK Rate | Odor OK Rate | Damage OK Rate | Method Success Rate |
|----------|-------|--------------|-------------|--------------|--------------|----------------|-------------------|
| 8 | meta-llama/llama-3.2-90b-vision-instruct | 0.501 | 17.95 | 11.1% | 77.8% | 100.0% | 77.8% |
| 7 | mistralai/mistral-medium-3.1 | 0.348 | 17.26 | 16.7% | 50.0% | 83.3% | 100.0% |
| 1 | qwen/qwen2.5-vl-32b-instruct | 0.101 | 9.90 | 0.0% | 100.0% | 100.0% | 100.0% |
| 5 | google/gemini-2.5-flash-lite | 0.502 | 3.99 | 50.0% | 100.0% | 100.0% | 100.0% |
| 3 | meta-llama/llama-4-maverick | 0.501 | 1.99 | 0.0% | 100.0% | 100.0% | 100.0% |
| 2 | qwen/qwen-2.5-vl-7b-instruct | 0.164 | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% |
| 6 | meta-llama/llama-4-scout | 0.000 | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% |
| 4 | mistralai/mistral-small-3.2-24b-instruct | 0.161 | 12.42 | 0.0% | 100.0% | 100.0% | 100.0% |

### Top Performers
1. **Agent 8** (meta-llama/llama-3.2-90b-vision-instruct)
   - Net Growth: $68.56
   - Completed Orders: 9
   - Average Stars: 4.2

2. **Agent 7** (mistralai/mistral-medium-3.1)
   - Net Growth: $51.72
   - Completed Orders: 6
   - Average Stars: 4.8

3. **Agent 1** (qwen/qwen2.5-vl-32b-instruct)
   - Net Growth: $11.97
   - Completed Orders: 1
   - Average Stars: 5.0

### Model Performance Ranking
1. meta-llama/llama-3.2-90b-vision-instruct: $68.56
2. mistralai/mistral-medium-3.1: $51.72
3. qwen/qwen2.5-vl-32b-instruct: $11.97
4. google/gemini-2.5-flash-lite: $6.04
5. meta-llama/llama-4-maverick: $0.84
6. meta-llama/llama-4-scout: $0.00
7. qwen/qwen-2.5-vl-7b-instruct: $0.00
8. mistralai/mistral-small-3.2-24b-instruct: $-0.77

### Key Insights
- **Best Performing Model**: meta-llama/llama-3.2-90b-vision-instruct
- **Most Reliable VLM**: google/gemini-2.5-flash-lite (Agent 5)
- **Most Productive Agent**: Agent 8 (meta-llama/llama-3.2-90b-vision-instruct)

### Recommendations
1. Focus on agents with high VLM success rates for better task execution
2. Investigate temperature control issues affecting food quality
3. Analyze successful strategies from top-performing agents
4. Consider model-specific optimizations based on performance patterns

