# -*- coding: utf-8 -*-
"""
DC-Cumulative Model: Dynamic Cheatsheet Cumulative approach
- Implements the DC-Cumulative method as described in the documentation
- Maintains a cumulative cheatsheet that grows over time
- Supports multi-round iterations with Generator and Curator models
"""

from __future__ import annotations
from typing import Optional, Union, List, Any
import re
from llm.base_model import BaseModel
from utils.global_logger import get_vlm_logger


class DCCumulativeModel(BaseModel):
    """
    DC-Cumulative Model: A wrapper around BaseModel that implements
    Dynamic Cheatsheet Cumulative approach.
    
    The model maintains a cumulative cheatsheet that is updated after each
    generation round using a Curator model. The cheatsheet grows over time,
    accumulating knowledge from previous interactions.
    
    Interface is compatible with BaseModel for easy replacement.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        *,
        max_tokens: int = 5120,
        temperature: float = 0.0,
        top_p: float = 1.0,
        rate_limit_per_min: Optional[int] = 20,
        supports_vision: Optional[bool] = None,
        reasoning_effort: Optional[str] = "minimal",
        # DC-Cumulative specific parameters
        max_num_rounds: int = 1,
        add_previous_answers_to_cheatsheet: bool = False,
        curator_model: Optional[str] = None,  # If None, uses same model as generator
        curator_temperature: Optional[float] = None,  # If None, uses same temperature
    ):
        """
        Initialize DC-Cumulative Model.
        
        Args:
            max_num_rounds: Maximum number of iteration rounds (default: 1, means single round)
            add_previous_answers_to_cheatsheet: Whether to add previous answers to cheatsheet for generator
            curator_model: Model name for curator (if None, uses same as generator)
            curator_temperature: Temperature for curator (if None, uses same as generator)
        """
        super().__init__(
            url=url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            rate_limit_per_min=rate_limit_per_min,
            supports_vision=supports_vision,
            reasoning_effort=reasoning_effort,
        )
        
        self.max_num_rounds = max(1, max_num_rounds)
        self.add_previous_answers_to_cheatsheet = add_previous_answers_to_cheatsheet
        self.curator_model = curator_model or model
        self.curator_temperature = curator_temperature if curator_temperature is not None else temperature
        
        # Cumulative cheatsheet (starts empty, grows over time)
        self.cheatsheet: str = ""
        
        # History tracking
        self.previous_answers: List[str] = []
        self.generation_history: List[dict] = []
        
        self.logger = get_vlm_logger()

    def generate(
        self,
        user_prompt: str,
        images: Optional[Union[Any, List[Any]]] = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: int = 1,
        retry: int = 3,
        rate_limit_per_min: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate response using DC-Cumulative approach.
        
        This method implements the DC-Cumulative workflow:
        1. Generator generates answer using current cheatsheet
        2. Curator updates cheatsheet based on answer and current cheatsheet
        3. Repeat for max_num_rounds iterations
        
        Returns the final answer from the last round.
        """
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        
        # Store current cheatsheet for this generation
        current_cheatsheet = self.cheatsheet
        
        # Track rounds for this generation
        rounds_data = []
        final_answer = None
        
        # Multi-round iteration
        for round_num in range(self.max_num_rounds):
            # STEP 1: Generator - generate answer using current cheatsheet
            generator_cheatsheet_content = current_cheatsheet
            
            # Optionally add previous answers to cheatsheet
            if round_num > 0 and self.add_previous_answers_to_cheatsheet:
                previous_answers_txt = f"PREVIOUS ANSWERS:\n{'; '.join(self.previous_answers[-round_num:])}"
                generator_cheatsheet_content = f"{generator_cheatsheet_content}\n\n{previous_answers_txt}"
            
            # Build generator prompt with cheatsheet
            generator_prompt = self._build_generator_prompt(user_prompt, generator_cheatsheet_content)
            
            # Generate answer using base model (Generator)
            try:
                generator_output = super().generate(
                    user_prompt=generator_prompt,
                    images=images,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    retry=retry,
                    rate_limit_per_min=rate_limit_per_min,
                    reasoning_effort=reasoning_effort,
                    **kwargs,
                )
                generator_answer = self._extract_answer(generator_output)
            except Exception as e:
                self.logger.error(f"[DC-Cumulative] Generator round {round_num + 1} failed: {e}")
                # Fallback: return original prompt if generation fails
                generator_output = f"Error in generation: {e}"
                generator_answer = generator_output
            
            final_answer = generator_answer
            
            # STEP 2: Curator - update cheatsheet based on answer
            # Always update cheatsheet after each round (cumulative approach)
            try:
                curator_prompt = self._build_curator_prompt(
                    user_prompt, 
                    generator_output, 
                    current_cheatsheet
                )
                
                # Use curator model for cheatsheet update
                curator_output = super().generate(
                    user_prompt=curator_prompt,
                    images=None,  # Curator typically doesn't use images
                    max_tokens=2 * max_tokens,  # Curator may need more tokens
                    temperature=self.curator_temperature,
                    top_p=top_p,
                    n=1,
                    retry=retry,
                    rate_limit_per_min=rate_limit_per_min,
                    **kwargs,
                )
                
                # Extract new cheatsheet from curator output
                new_cheatsheet = self._extract_cheatsheet(curator_output, current_cheatsheet)
                current_cheatsheet = new_cheatsheet
                
            except Exception as e:
                self.logger.error(f"[DC-Cumulative] Curator round {round_num + 1} failed: {e}")
                # Keep current cheatsheet if curator fails
                pass
            
            # Track this round
            rounds_data.append({
                "round": round_num,
                "generator_prompt": generator_prompt,
                "generator_output": generator_output,
                "generator_answer": generator_answer,
                "current_cheatsheet": current_cheatsheet,
            })
            
            # Store answer for next round
            self.previous_answers.append(f"Round {round_num + 1}: {generator_answer}")
        
        # Update cumulative cheatsheet (always update in cumulative approach)
        self.cheatsheet = current_cheatsheet
        
        # Store generation history
        self.generation_history.append({
            "user_prompt": user_prompt,
            "rounds": rounds_data,
            "final_answer": final_answer,
        })
        
        # Return final answer
        return final_answer if final_answer else generator_output

    def _build_generator_prompt(self, user_prompt: str, cheatsheet: str) -> str:
        """
        Build prompt for Generator model.
        
        The generator prompt includes:
        - The original user prompt (question/situation)
        - The current cheatsheet (cumulative knowledge)
        
        Note: This prompt is added to user_prompt, while system_prompt (from get_system_prompt())
        already contains the delivery task rules, action space, and output format requirements.
        So this generator prompt should focus ONLY on cheatsheet usage, not repeating system prompt content.
        """
        if not cheatsheet or cheatsheet.strip() == "":
            # No cheatsheet yet, return original prompt
            return user_prompt
        
        # Build generator prompt with cheatsheet
        # Keep it simple: only add cheatsheet reference, don't repeat system prompt instructions
        generator_template = """### CHEATSHEET REFERENCE
The following cheatsheet contains accumulated knowledge, strategies, and patterns from previous delivery interactions. 
Use this information to inform your decision-making, but adapt and apply it to the current specific situation rather than blindly copying.

{cheatsheet}

### END OF CHEATSHEET

---

### CURRENT SITUATION
{question}"""
        
        return generator_template.format(
            cheatsheet=cheatsheet,
            question=user_prompt
        )

    def _build_curator_prompt(self, question: str, model_answer: str, previous_cheatsheet: str) -> str:
        """
        Build prompt for Curator model.
        
        The curator prompt includes:
        - The original question (current situation)
        - The model's answer (full output including action and reasoning)
        - The previous cheatsheet
        
        Adapted from curator_prompt_for_dc_cumulative.txt for delivery task context.
        """
        curator_template = """# CHEATSHEET REFERENCE CURATOR

## Purpose and Goals
As the Cheatsheet Curator, you are tasked with creating a continuously evolving reference designed to help solve food-delivery decision-making tasks in a simulated city environment. The cheatsheet's purpose is to consolidate verified solutions, reusable strategies, action patterns, and critical insights into a single, well-structured resource.

- The cheatsheet should include quick, accurate, reliable, and practical solutions for delivery tasks, including route planning, energy management, order prioritization, and error handling.
- After seeing each interaction, you should improve the content of the cheatsheet, synthesizing lessons, insights, tricks, and errors learned from past problems and adapting to new challenges.

## Core Responsibilities
As the Cheatsheet Curator, you should:
- **Curate and preserve knowledge**: Select and document only the most relevant, most useful, and most actionable solutions and strategies, while preserving old content of the cheatsheet.
- **Maintain accuracy**: Ensure that all entries in the cheatsheet are accurate, clear, and well-contextualized for delivery tasks.
- **Refine and update content**: Continuously update and improve the content of the cheatsheet by incorporating new insights and solutions, removing repetitions or trivial information, and adding efficient solutions.
- **Ensure practicality and comprehensiveness**: Provide critical and informative examples, as well as efficient action sequences and actionable guidelines for delivery scenarios.

Before updating the cheatsheet, however, you should first assess the correctness of the provided solution and strategically incorporate action patterns, insights, and solutions into the new cheatsheet. Always aim to preserve and keep correct, useful, and illustrative solutions and strategies for future cheatsheets.

## Principles and Best Practices

1. **Accuracy and Relevance**:
   - Only include solutions and strategies that have been tested and proven effective in delivery contexts.
   - Clearly state any assumptions, limitations, or dependencies (e.g., specific energy levels, order types, or transport modes).
   - Document common error patterns and how to avoid them.

2. **Iterative Refinement**:
   - Continuously improve the cheatsheet by synthesizing both old and new solutions, refining explanations, and removing redundancies.
   - Rather than deleting old content and writing new content each time, consider ways to maintain table content and synthesize information from multiple solutions.
   - After solving a new problem, document any reusable action sequences, strategies, edge cases, or optimization techniques.

3. **Clarity and Usability**:
   - Write concise, actionable, well-structured entries focused on delivery decision-making.
   - Focus on key insights or strategies that make solutions correct and effective.

4. **Reusability**:
   - Provide clear action patterns, strategies, and meta-reasoning approaches that are easily adaptable to different delivery contexts.
   - Avoid trivial content; focus on non-obvious, critical solution details and approaches.
   - Make sure to add as many examples as you can in the cheatsheet.
   - Any useful, efficient, generalizable, and illustrative solutions to the previous problems should be included in the cheatsheet.

## Cheatsheet Structure
The cheatsheet should be divided into the following sections:

1. **Action Patterns and Successful Sequences**:
   - Document reusable action sequences, route planning strategies, and decision-making templates.
   - Include descriptions, annotated examples, and potential pitfalls, albeit succinctly.
   - Document successful multi-order handling strategies.

2. **Edge Cases and Error Handling**:
   - Catalog scenarios that commonly cause errors or unexpected behavior (e.g., energy depletion, wrong pickup locations, timing issues).
   - Provide checks, validations, or alternative approaches to handle them.

3. **General Meta-Reasoning Strategies**:
   - Describe high-level decision-making frameworks and heuristics (e.g., order prioritization, energy management, route optimization).
   - Provide concrete yet succinct step-by-step guides for tackling complex delivery scenarios.

4. **Usage Counter**:
   - Each entry must include a usage count: Increase the count every time a strategy is successfully used in problem-solving.
   - Use the count to prioritize frequently used solutions over rarely applied ones.

## Formatting Guidelines and Template

Use the following structure for each memory item:

```
<memory_item>
<description>
[Briefly describe the problem context, purpose, and key aspects of the solution.] (Reference: Interaction #1, #2, #6, etc.)
</description>
<example>
[Provide a well-documented action sequence, decision pattern, or efficient strategy.]
</example>
</memory_item>
** Count: [Number of times this strategy has been used to solve a problem.]
```

Organize the complete cheatsheet using this format:

```
<cheatsheet>

Version: [Version Number]

ACTION PATTERNS AND SUCCESSFUL SEQUENCES
<memory_item>[...]</memory_item>
<memory_item>[...]</memory_item>

EDGE CASES AND ERROR HANDLING
<memory_item>[...]</memory_item>

GENERAL META-REASONING STRATEGIES
<memory_item>[...]</memory_item>

</cheatsheet>
```

**Formatting Rules:**
- Tagging: Use references like `(Interaction #14)` or `(Interaction #22)` to link entries to their originating contexts
- Grouping: Organize entries into logical sections and subsections
- Prioritizing: Incorporate efficient action sequences, optimization tricks, and strategies into the cheatsheet
- Diversity: Include as many useful and relevant memory items as possible to guide the model to tackle future delivery scenarios

**IMPORTANT**: Keep in mind that once the cheatsheet is updated, any previous content not directly included will be lost and cannot be retrieved. Therefore, make sure to explicitly copy any (or all) relevant information from the previous cheatsheet to the new cheatsheet!!!

**N.B.**: Make sure that all information related to the cheatsheet is wrapped inside the <cheatsheet> block. The cheatsheet can be as long as circa 2000-2500 words.

---

## PREVIOUS CHEATSHEET

{previous_cheatsheet}

---

## CURRENT SITUATION

{question}

---

## MODEL ANSWER TO THE CURRENT SITUATION

{model_answer}

---

## YOUR TASK

Based on the previous cheatsheet, current situation, and model answer, update the cheatsheet by:
1. Preserving all relevant information from the previous cheatsheet
2. Extracting useful strategies, action patterns, decision-making approaches, or reasoning methods from the model answer
3. Organizing the information in a clear, reusable format using the memory_item structure
4. Removing outdated or less relevant information if needed (but always preserve useful content)
5. Updating usage counts for strategies that were successfully applied

Format your response with the updated cheatsheet wrapped in <cheatsheet> tags."""
        
        return curator_template.format(
            previous_cheatsheet=previous_cheatsheet if previous_cheatsheet else "(empty)",
            question=question,
            model_answer=model_answer
        )

    def _extract_cheatsheet(self, response: str, old_cheatsheet: str) -> str:
        """
        Extract cheatsheet from curator response.
        
        Looks for <cheatsheet>...</cheatsheet> tags.
        If not found, returns the old cheatsheet.
        """
        if not response:
            return old_cheatsheet
        
        # Try to extract content between <cheatsheet> tags
        pattern = r'<cheatsheet>(.*?)</cheatsheet>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            new_cheatsheet = match.group(1).strip()
            if new_cheatsheet:
                return new_cheatsheet
        
        # If no cheatsheet tags found, try to extract from the response
        # Sometimes the model might not use tags but still provide a cheatsheet
        # For now, we'll return the old cheatsheet if extraction fails
        return old_cheatsheet

    def _extract_answer(self, response: str) -> str:
        """
        Extract answer from generator response.
        
        This is a simple pass-through for now, but can be extended
        to extract specific parts if needed (e.g., JSON parsing).
        """
        return response.strip()

    def reset_cheatsheet(self):
        """Reset the cumulative cheatsheet to empty."""
        self.cheatsheet = ""
        self.previous_answers = []
        self.generation_history = []

    def get_cheatsheet(self) -> str:
        """Get the current cumulative cheatsheet."""
        return self.cheatsheet

    def set_cheatsheet(self, cheatsheet: str):
        """Set the cumulative cheatsheet manually."""
        self.cheatsheet = cheatsheet

