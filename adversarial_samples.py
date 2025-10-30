import click
import requests
import random
import os
from typing import List, Set, Optional, Dict, Tuple


# --- GRAPHEME AUGMENTER CLASS (As provided by the user) ---
class GraphemeAugmenter:
    """
    Implements the GraphemeAug algorithm for generating hard negative (confusable)
    keyword spotting examples based on single-grapheme edits (Insertion, Deletion,
    Substitution).
    """
    # Standard English Vowels and Consonants (Case-insensitive)
    STANDARD_VOWELS = set('aeiou')
    STANDARD_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')

    def __init__(self, max_edits: int = 1, min_edits: int = 1, language_vowels: Optional[Set[str]] = None,
                 language_consonants: Optional[Set[str]] = None):
        if max_edits < 1:
            raise ValueError("max_edits must be 1 or greater.")
        if min_edits < 0:
            raise ValueError("min_edits cannot be negative.")
        if min_edits > max_edits:
            raise ValueError("min_edits cannot be greater than max_edits.")

        self.max_edits = max_edits
        self.min_edits = min_edits

        self.VOWELS = self.STANDARD_VOWELS.union({c.lower() for c in (language_vowels or set())})
        self.CONSONANTS = self.STANDARD_CONSONANTS.union({c.lower() for c in (language_consonants or set())})

        self.ALL_GRAPHEMES = self.VOWELS.union(self.CONSONANTS)

        if not self.VOWELS.isdisjoint(self.CONSONANTS):
            print(
                f"Warning: Overlap found between VOWELS and CONSONANTS: {self.VOWELS.intersection(self.CONSONANTS)}. This may affect substitution rules.")

    def _get_char_class(self, char: str) -> Optional[str]:
        char_lower = char.lower()
        if char_lower in self.VOWELS:
            return 'vowel'
        elif char_lower in self.CONSONANTS:
            return 'consonant'
        return None

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        # Standard Levenshtein implementation (omitted for brevity)
        if len(s1) < len(s2):
            return GraphemeAugmenter._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, char1 in enumerate(s1):
            current_row = [i + 1]
            for j, char2 in enumerate(s2):
                cost = 0 if char1 == char2 else 1
                current_row.append(min(
                    previous_row[j + 1] + 1,
                    current_row[j] + 1,
                    previous_row[j] + cost
                ))
            previous_row = current_row
        return previous_row[-1]

    # --- Core Edit Operation (Random Single Edit) ---
    def _get_random_one_edit(self, text: str) -> str:
        if not text:
            return random.choice(list(self.ALL_GRAPHEMES))

        edit_type = random.randint(0, 2)

        if edit_type == 0:  # Insertion
            pos = random.randint(0, len(text))
            grapheme = random.choice(list(self.ALL_GRAPHEMES))
            return text[:pos] + grapheme + text[pos:]

        elif edit_type == 1:  # Deletion
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1:]

        else:  # Substitution
            sub_attempts = 0
            while sub_attempts < 10:
                pos = random.randint(0, len(text) - 1)
                char_to_replace = text[pos]
                char_class = self._get_char_class(char_to_replace)

                if char_class == 'vowel':
                    target_set = self.VOWELS
                elif char_class == 'consonant':
                    target_set = self.CONSONANTS
                else:
                    sub_attempts += 1
                    continue

                possible_replacements = list(target_set - {char_to_replace.lower()})

                if possible_replacements:
                    replacement_grapheme = random.choice(possible_replacements)
                    return text[:pos] + replacement_grapheme + text[pos + 1:]

                sub_attempts += 1

            return self._get_random_one_edit(text)

    def generate_confusables(self, keyword: str, n_samples: Optional[int] = None) -> List[str]:
        original_keyword = keyword.lower()
        generated_confusables: Set[str] = set()

        if n_samples is not None:
            if n_samples == 0:
                return []

            seed_pool = {original_keyword}
            attempts = 0
            max_attempts = n_samples * 10 + 100  # Add a buffer

            while len(generated_confusables) < n_samples and attempts < max_attempts:
                word_to_mutate = random.choice(list(seed_pool))
                candidate = self._get_random_one_edit(word_to_mutate)
                distance = self._levenshtein_distance(original_keyword, candidate)

                if (self.min_edits <= distance <= self.max_edits and
                        candidate != original_keyword and
                        candidate not in generated_confusables):

                    generated_confusables.add(candidate)
                    if distance < self.max_edits:
                        seed_pool.add(candidate)

                elif distance < self.max_edits:
                    # Allow mutations that are closer than min_edits to be used for next mutation
                    # but only if they are not the original keyword.
                    if candidate != original_keyword:
                        seed_pool.add(candidate)

                attempts += 1

        else:
            # Original exhaustive method (can generate too many samples)
            # This is typically not used with n_samples constraint
            raise NotImplementedError("Exhaustive generation is not supported in hybrid mode.")

        return sorted(list(generated_confusables))


# --- LLM PROMPT DEFINITIONS ---
SYSTEM = """You are an expert in computational linguistics, phonetics, and adversarial machine learning. Your core function is to generate lists of strings that are acoustically and linguistically similar to a given wake word or keyword. These generated strings serve as **adversarial samples** intended to trick an Automatic Speech Recognition (ASR) system or a keyword spotter.

**Generation Goal:** The adversarial samples must primarily **rhyme** with the components of the input keyword or mimic its overall rhythm and length.

**Output Constraints (CRITICAL):**
1.  **Format:** Output only the generated adversarial strings.
2.  **Delimiter:** Use a **newline** character to separate each sample (one sample per line).
3.  **Exclusion:** DO NOT include any introductory text, numbering, bullet points, explanations, or concluding remarks. The output must be the raw list of samples.
4.  **Diversity:** all samples MUST be unique and PHONETICALLY similar.

**Example:**
[Input Keyword]: "hey computer"
[Target Output]:
say scooter
pay intruder
gray maneuver
stay pewter"""

USER = """**Generate {n_samples} adversarial, rhyming samples for the following keyword:** {word}"""


def generate_llm_samples(url: str, model: str, wakeword: str, n_samples: int) -> List[str]:
    """Generates adversarial samples using the LLM API."""
    if n_samples <= 0:
        return []

    click.echo(f"  🚀 LLM (Target: {n_samples}):", nl=False)
    try:
        # Request for all samples in one call, and then rely on LLM for diversity.
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": USER.format(word=wakeword, n_samples=n_samples * 2),  # Request more than needed for filtering
                "system": SYSTEM,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()

        # Process response
        result = response.json()["response"].strip().split("\n")

        # Filter results based on word count
        word_count = len(wakeword.split())
        filtered_result = [
            r.strip() for r in result
            if len(r.strip().split()) == word_count
        ]

        # Return up to n_samples unique results
        return list(set(filtered_result))[:n_samples]

    except requests.exceptions.RequestException as e:
        click.echo(f" [API Error: {e}]", err=True)
        return []


def generate_grapheme_samples(augmenter: GraphemeAugmenter, wakeword: str, n_samples: int) -> List[str]:
    """Generates adversarial samples using the GraphemeAug algorithm."""
    if n_samples <= 0:
        return []

    click.echo(f"  🧬 GraphemeAug (Target: {n_samples}):", nl=False)
    try:
        # n_samples controls the random generation count
        new_samples = augmenter.generate_confusables(wakeword, n_samples=n_samples)
        return new_samples
    except Exception as e:
        click.echo(f" [GraphemeAug Error: {e}]", err=True)
        return []


def resolve_output_mode(output_file: str, file_mode: str) -> Optional[str]:
    """
    Checks for file existence and determines the correct file open mode.
    Raises a ClickException or returns the appropriate Python file mode ('w' or 'a').
    """
    file_exists = os.path.exists(output_file)

    if file_exists:
        if file_mode == 'error':
            raise click.ClickException(
                f"❌ Output file already exists: '{output_file}'. Use '--file-mode overwrite' or '--file-mode append'.")
        elif file_mode == 'overwrite':
            click.echo(f"⚠️ **Overwriting** existing file: '{output_file}'")
            return 'w'
        elif file_mode == 'append':
            click.echo(f"📝 **Appending** to existing file: '{output_file}'")
            return 'a'

    # If file doesn't exist, or file_mode is 'overwrite', we use 'w'
    click.echo(f"📝 Creating new output file: '{output_file}'")
    return 'w'


@click.command()
@click.option('-o', '--output-file', required=True,
              help='The output .txt file to write all unique adversarial samples to (one per line).')
@click.option('-F', '--file-mode', type=click.Choice(['append', 'overwrite', 'error'], case_sensitive=False),
              default='append',
              help='Behavior if the output file already exists. Defaults to append.')
@click.option('-u', '--url', required=True,
              help='**REQUIRED.** Base URL for the LLM API (e.g., http://100.88.41.41:11434).')
@click.option('-m', '--model', required=True,
              help='**REQUIRED.** The LLM model to use for generation (e.g., gemma3:4b).')
@click.option('--grapheme-max-edits', default=3, type=int, help='Max Levenshtein distance for GraphemeAug.')
@click.option('--grapheme-min-edits', default=2, type=int, help='Min Levenshtein distance for GraphemeAug.')
@click.option('-w', '--wakeword', 'wakewords', multiple=True, required=True,
              help='**REQUIRED.** One or more wake words to generate samples for (can be specified multiple times).')
@click.option('-n', '--n-samples', required=True, type=int,
              help='**REQUIRED.** The **total** target number of unique samples to collect/generate per run for each wake word.')
@click.option('--llm-weight', default=1, type=float, help='Weight for LLM generation. 0 to skip. 1 for max.')
@click.option('--grapheme-weight', default=0.2, type=float,
              help='Weight for GraphemeAug generation. 0 to skip. 1 for max.')
def generate_samples(output_file: str, file_mode: str, url: str, model: str, grapheme_max_edits: int,
                     grapheme_min_edits: int, wakewords: List[str], n_samples: int, llm_weight: float,
                     grapheme_weight: float):
    """
    Generates adversarial samples using a weighted hybrid approach combining
    LLM (rhyming) and GraphemeAug (edit distance) methods, and outputs them
    to a .txt file, respecting the chosen file-mode.
    """

    # --- FILE EXISTENCE CHECK AND MODE RESOLUTION ---
    file_open_mode = resolve_output_mode(output_file, file_mode)

    # In 'append' mode, we need to load existing samples to ensure uniqueness
    existing_samples: Set[str] = set()
    if file_open_mode == 'a' and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                # Read existing lines, strip whitespace, and add to set
                existing_samples.update({line.strip() for line in f if line.strip()})
            click.echo(f"Loaded {len(existing_samples)} existing unique samples from '{output_file}'.")
        except Exception as e:
            raise click.ClickException(f"Error reading existing file for append mode: {e}")

    # --- INPUT VALIDATION AND SETUP ---
    total_weight = llm_weight + grapheme_weight
    if total_weight <= 0:
        click.echo("❌ Error: Both LLM and GraphemeAug weights are 0. No generation will occur.", err=True)
        return

    # Normalize weights and calculate target sample counts
    llm_ratio = llm_weight / total_weight
    grapheme_ratio = grapheme_weight / total_weight

    llm_n_samples = int(n_samples * llm_ratio)
    grapheme_n_samples = int(n_samples * grapheme_ratio)

    # Adjust for rounding if the sum is less than n_samples
    if llm_n_samples + grapheme_n_samples < n_samples:
        if llm_ratio >= grapheme_ratio:
            llm_n_samples += (n_samples - (llm_n_samples + grapheme_n_samples))
        else:
            grapheme_n_samples += (n_samples - (llm_n_samples + grapheme_n_samples))

    click.echo(f"✨ **Hybrid Generation** (Total Target: {n_samples})")
    click.echo(f"  LLM Target: {llm_n_samples} samples (Weight: {llm_weight})")
    click.echo(f"  GraphemeAug Target: {grapheme_n_samples} samples (Weight: {grapheme_weight})")
    click.echo("---")

    # Initialize GraphemeAugmenter
    augmenter = None
    if grapheme_weight > 0:
        try:
            augmenter = GraphemeAugmenter(max_edits=grapheme_max_edits, min_edits=grapheme_min_edits)
            click.echo(f"🧬 GraphemeAug initialized (Max Edits: {grapheme_max_edits}, Min Edits: {grapheme_min_edits}).")
        except ValueError as e:
            click.echo(f"❌ Error initializing GraphemeAug: {e}", err=True)
            grapheme_n_samples = 0  # Disable if initialization fails

    # Check LLM connection settings
    if llm_weight > 0:
        click.echo(f"🚀 LLM initialized at: {url} (Model: {model}).")

    if llm_n_samples + grapheme_n_samples == 0:
        click.echo("❌ No generation methods are active after checks.", err=True)
        return

    # Store all unique generated samples for all wake words, including existing ones if in append mode
    all_adversarial_samples: Set[str] = existing_samples.copy()
    initial_count = len(existing_samples)

    # Process each wake word
    for wakeword in wakewords:
        click.echo(f"\n--- Processing '{wakeword}' ---")
        newly_added_count_for_word = 0

        # 1. LLM Generation
        if llm_n_samples > 0:
            new_llm_samples = generate_llm_samples(url, model, wakeword, llm_n_samples)
            current_added = 0
            for r in new_llm_samples:
                r_clean = r.strip()
                # Check for uniqueness against the entire collected set
                if r_clean and r_clean not in all_adversarial_samples:
                    all_adversarial_samples.add(r_clean)
                    current_added += 1
            newly_added_count_for_word += current_added
            click.echo(f" Added {current_added} new unique LLM samples.")

        # 2. GraphemeAug Generation
        if grapheme_n_samples > 0 and augmenter:
            new_grapheme_samples = generate_grapheme_samples(augmenter, wakeword, grapheme_n_samples)
            current_added = 0
            for r in new_grapheme_samples:
                r_clean = r.strip()
                # Check for uniqueness against the entire collected set
                if r_clean and r_clean not in all_adversarial_samples:
                    all_adversarial_samples.add(r_clean)
                    current_added += 1
            newly_added_count_for_word += current_added
            click.echo(f" Added {current_added} new unique GraphemeAug samples.")

        click.echo(f"  Total unique samples so far: {len(all_adversarial_samples)}.")

    # --- OUTPUT TO FILE ---
    # Only write the *newly added* samples if in append mode,
    # or write *all* collected samples if in overwrite/new file mode.
    if file_open_mode == 'a':
        samples_to_write = sorted(list(all_adversarial_samples - existing_samples))
        total_unique_count = len(all_adversarial_samples)
        written_count = len(samples_to_write)
        if written_count > 0:
            # Ensure a leading newline if appending to a non-empty file
            content_to_write = '\n' + '\n'.join(samples_to_write)
        else:
            content_to_write = ''

    else:  # 'w' mode (overwrite or new file)
        samples_to_write = sorted(list(all_adversarial_samples))
        total_unique_count = len(samples_to_write)
        written_count = total_unique_count
        content_to_write = '\n'.join(samples_to_write)

    try:
        if content_to_write:
            with open(output_file, file_open_mode) as f:
                f.write(content_to_write)
        elif file_open_mode == 'w':  # If file is new/overwritten but no content, write empty file
            with open(output_file, 'w') as f:
                pass

        click.echo("\n" + "=" * 50)
        if file_open_mode == 'a':
            click.echo(f"✅ Appended **{written_count}** new unique samples to: **{output_file}**")
            click.echo(f"   (Total unique samples in file: {total_unique_count})")
        else:
            click.echo(f"✅ All generated samples (**{written_count}** total unique) written to: **{output_file}**")
        click.echo("=" * 50)

    except Exception as e:
        click.echo(f"❌ Error writing to output file '{output_file}': {e}", err=True)


if __name__ == '__main__':
    # Add a check for no arguments to show help message
    if len(os.sys.argv) == 1:
        generate_samples.main(['--help'])
    else:
        generate_samples()
