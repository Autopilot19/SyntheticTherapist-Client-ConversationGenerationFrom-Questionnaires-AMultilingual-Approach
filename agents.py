import random
import re
import os
from time import time
from datetime import datetime
from typing import Optional

from langchain_core.prompts import PromptTemplate
from llms import LLM, LLama, Qwen, Gemma, Mistral, Command
from loguru import logger

SET_NAME = os.getenv("SET_NAME", "blank")

lang= "de" 

THERAPIST_TAGS = [
    "Therapeut",     # DE 
    "Therapist",     # EN
    "Thérapeute",    # FR
    "Therapiste",    # FR (optional)
    "المعالج",       # AR
    "المعالجة",      # AR
]

CLIENT_TAGS = [
    "Klient",        # DE 
    "Client",        # EN/FR common
    "العميل",       # AR
    "العميلة",      # AR
]

class LLMsType:
    @staticmethod
    def get_llm(llm) -> LLama | Qwen | Gemma | Mistral | Command :
        """Get the LLM type based on the input string."""
        match llm:
            case "command":
                return Command()
            case "mistral":
                return Mistral()
            case "llama3":
                return LLama("llama3")
            case "nemotron":
                return LLama("nemotron")
            case "qwen_qwq":
                return Qwen("qwen_qwq")
            case "qwen-2.5":
                return Qwen("qwen-2.5")
            case "gemma":
                return Gemma()
            case _:
                raise ValueError(f"Unsupported LLM type: {llm}")

    @staticmethod
    def get_llm_token(llm) -> str:
        match llm:
            case "command":
                return "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
            case "mistral":
                return "[/INST]"
            case "gemma":
                return """<end_of_turn>
            <start_of_turn>model
            """
            case ("llama3" | "nemotron"):
                return """<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            case llm if "qwen" in llm:
                return "<|im_end|>"
            case _:
                raise ValueError(f"Unsupported LLM type: {llm}")


def heavy_filter(response: str) -> str:
    if '[END]' in response:
        response = response.replace("[END]", "[/END]")
    if '[**/END**]' in response:
        response = response.replace("[**/END**]", "[/END]")
    if '[**END**]' in response:
        response = response.replace("[**END**]", "[/END]")
    if '/END' in response:
        response = response.replace("/END", "[/END]")
    if '[[/END]]' in response:
        response = response.replace("[[/END]]", "[/END]")
    if '[]' in response:
        response = response.replace("[]", "[/END]")

    return response


class Agent:
    def __init__(
        self, llm: str, prompt_text: str = "", questionnaire: str = ""
    ) -> None:
        self.llm_type: str = llm
        self.llm: LLM = LLMsType.get_llm(llm)
        self.llm_token: str = LLMsType.get_llm_token(llm)
        self.prompt_text = self.load_prompt(prompt_text)
        self.questionnaire_name = questionnaire.split("/")[-1].split(".")[0]
        self.questionnaire_style = questionnaire.split("/")[-3]
        self.questionnaire = self.load_questionnaire(questionnaire)
        self.prompt_template = PromptTemplate(
            input_variables=["history", "questionnaire"], template=self.prompt_text
        )

    @staticmethod
    def load_prompt(file_name: str) -> str:
        with open(f"{file_name}", "r") as f:
            return f.read()

    @staticmethod
    def load_questionnaire(file_name: str) -> str:
        with open(f"{file_name}", "r") as f:
            return f.read()

    def generate(self, history: list[dict], comment: str = "") -> str:
        prompt: str = self.prompt_template.format(
            questionnaire=self.questionnaire,
            history="\n".join(f"{message['response']}" for message in history),
        )

        if comment != "":
            if "qwen" not in self.llm_type:
                prompt = prompt.split(self.llm_token)[0]
            else:
                parts = prompt.rsplit(self.llm_token, 1)
                if len(parts) > 1:
                    prompt = parts[0] + parts[1]
                else:
                    raise ValueError("No token found in the prompt")
            prompt = f"{prompt}\n{comment}\n{self.llm_token}"
            logger.debug(f"{comment}")
        return self.llm.generate(prompt)

    def post_process(self, response: str, turn) -> str:
        return ""


class Client(Agent):
    def __init__(
        self, llm: str, prompt_text: str = "", questionnaire: str = ""
    ) -> None:
        super().__init__(llm, prompt_text, questionnaire)

    def post_process(self, response: str, turn: int) -> str:
        for tag in CLIENT_TAGS:
            response = response.replace(f"**{tag}:**", f"{tag}:")

        if "<|END_RESPONSE|>" in response:
            response = response.replace("<|END_RESPONSE|>", "")

        # Remove anything before the client's response
        if self.llm_type == "qwen_qwq":
            match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
            response = match.group(1).strip() if match else response

        tag_alt = "|".join(map(re.escape, CLIENT_TAGS))

        if self.llm_type == "qwen":
            # Remove agent <tool_call> token in qwen
            cleaned_response: str = response.replace("<tool_call>", "")
        else:
            cleaned_response = response

        # Try to extract the client utterance if it exists inside extra text
        m = re.search(rf"({tag_alt})\s*:\s*.+", cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        if m:
            cleaned_response = m.group(0).strip()

        # We only need the client part of the response, we shouldn't let agent generate the THERAPIST response
        redundant_gen_id = next(
            (
                cleaned_response.lower().find(f"{t.lower()}:")
                for t in THERAPIST_TAGS
                if cleaned_response.lower().find(f"{t.lower()}:") != -1
            ),
            -1,
        )
        if redundant_gen_id != -1:
            cleaned_response = cleaned_response[:redundant_gen_id]

        cleaned_response = heavy_filter(cleaned_response)

        if "mdd" in self.questionnaire_style:
            if turn < 12:
                # Avoid ending the session too early
                cleaned_response = cleaned_response.replace("[/END]", "")
                cleaned_response = cleaned_response.replace("[END]", "")
        else:
            if turn < 7:
                # Avoid ending the session too early
                cleaned_response = cleaned_response.replace("[/END]", "")
                cleaned_response = cleaned_response.replace("[END]", "")

        # Strip spaces and newlines
        cleaned_response = cleaned_response.strip()

        if self.llm_type == "qwen_qwq" or self.llm_type == "glm":
            if self.llm_type == "qwen_qwq" or self.llm_type == "glm":
                pattern = r'Client:.*?\"'
                match = re.search(pattern, cleaned_response, re.DOTALL)

                if match:
                    # Get the matched text but remove the trailing quote
                    cleaned_response = match.group(0)[:-1]
                    cleaned_response = cleaned_response.strip()
                else:
                    # If no match, try a different approach - find the first quote that ends a sentence
                    end_quote_position = cleaned_response.find('."')
                    if end_quote_position != -1:
                        cleaned_response = cleaned_response[:end_quote_position + 1].strip()
                    else:
                        cleaned_response = cleaned_response.strip()

            cleaned_response = re.sub(r'\s*\(\d+\s+words\)$', '', cleaned_response)

        # In case the client response is empty
        s = cleaned_response.strip()
        if s == "" or any(s == f"{tag}:" for tag in CLIENT_TAGS):
            if lang == "fr":
                cleaned_response = random.choice([
                    "Client: [Pas de réponse]",
                    "Client: [Pause et réfléchit]",
                    "Client: [Reste silencieux]",
                    "Client: [Silence]",
                    "Client: Je ne sais pas quoi dire",
                    "Client: Je ne sais pas",
                ])
            elif lang == "en":
                cleaned_response = random.choice([
                    "Client: [No reply]",
                    "Client: [Pause and thinking]",
                    "Client: [Keep silent]",
                    "Client: [Quiet]",
                    "Client: I don't know what to say",
                    "Client: I don't know",
                ])
            elif lang == "de":
                cleaned_response = random.choice([
                    "Klient: [Keine Antwort]",
                    "Klient: [Pause und denkt nach]",
                    "Klient: [Schweigt]",
                    "Klient: [Still]",
                    "Klient: Ich weiß nicht, was ich sagen soll",
                    "Klient: Ich weiß nicht",
                ])
            elif lang == "ar":
                cleaned_response = random.choice([
                    "العميل: [لا رد]",
                    "العميل: [وقفة قصيرة ويفكر]",
                    "العميل: [يلتزم الصمت]",
                    "العميل: [صمت]",
                    "العميل: لا أعرف ماذا أقول",
                    "العميل: لا أعرف",
                ])

        if any(cleaned_response.startswith(f"{tag}:") for tag in CLIENT_TAGS):
            return cleaned_response
        else:
            if lang == "fr":
                return random.choice([
                    "Client: [Pas de réponse]",
                    "Client: [Pause et réfléchit]",
                    "Client: [Reste silencieux]",
                    "Client: [Silence]",
                    "Client: Je ne sais pas quoi dire",
                    "Client: Je ne sais pas",
                ])
            elif lang == "en":
                return random.choice([
                    "Client: [No reply]",
                    "Client: [Pause and thinking]",
                    "Client: [Keep silent]",
                    "Client: [Quiet]",
                    "Client: I don't know what to say",
                    "Client: I don't know",
                ])
            elif lang == "de":
                return random.choice([
                    "Klient: [Keine Antwort]",
                    "Klient: [Pause und denkt nach]",
                    "Klient: [Schweigt]",
                    "Klient: [Still]",
                    "Klient: Ich weiß nicht, was ich sagen soll",
                    "Klient: Ich weiß nicht",
                ])
            elif lang == "ar":
                return random.choice([
                    "العميل: [لا رد]",
                    "العميل: [وقفة قصيرة ويفكر]",
                    "العميل: [يلتزم الصمت]",
                    "العميل: [صمت]",
                    "العميل: لا أعرف ماذا أقول",
                    "العميل: لا أعرف",
                ])
                                


class Therapist(Agent):
    def __init__(
        self, llm: str, prompt_text: str = "", questionnaire: str = ""
    ) -> None:
        super().__init__(llm, prompt_text, questionnaire)

    def post_process(self, response: str, turn: int) -> str:
        for t in THERAPIST_TAGS:
            if f"**{t}:**" in response:
                response = response.replace(f"**{t}:**", f"{t}:")

        if "<|END_RESPONSE|>" in response:
            response = response.replace("<|END_RESPONSE|>", "")

        # Remove anything before the therapist response
        if self.llm_type == "qwen_qwq":
            match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
            response = match.group(1).strip() if match else response

        tag_alt = "|".join(map(re.escape, THERAPIST_TAGS))
        match = re.search(rf"(({tag_alt})\s*:\s*.+)", response, flags=re.IGNORECASE)

        if match:
            response = match.group(1).strip()

        # We only need the Therapist part of the response, we shouldn't let agent generate the client response
        redundant_gen_id = next((response.lower().find(f"{t.lower()}:") for t in CLIENT_TAGS if response.lower().find(f"{t.lower()}:") != -1), -1)
        if redundant_gen_id != -1:  # If the word is found
            response = response[:redundant_gen_id]

        if self.llm_type == "qwen":
            # Remove agent <tool_call> token in qwen
            response = response.replace("<tool_call>", "")

        response = heavy_filter(response)

        if "mdd" in self.questionnaire_style:
            if turn < 12:
                # Avoid ending the session too early
                response = response.replace("[/END]", "")
        else:
            if turn < 7:
                # Avoid ending the session too early
                response = response.replace("[/END]", "")

        # Strip spaces and newlines
        response = response.strip()

        if self.llm_type == "qwen_qwq" or self.llm_type == "glm":
            pattern = r'Therapist:.*?\"'
            match = re.search(pattern, response, re.DOTALL)

            if match:
                # Get the matched text but remove the trailing quote
                response = match.group(0)[:-1]  # Remove the last character (quote)
                response = response.strip()
            else:
                # If no match, try a different approach - find the first quote that ends a sentence
                end_quote_position = response.find('."')
                if end_quote_position != -1:
                    # Return everything up to and including the period but not the quote
                    response = response[:end_quote_position + 1].strip()
                else:
                    # Return the original text if no pattern is found
                    response = response.strip()

            response = re.sub(r'\s*\(\d+\s+words\)$', '', response)

        return response


class TherapySession:
    def __init__(
        self, client: Client, therapist: Therapist
    ) -> None:
        self.client: Client = client
        self.therapist: Therapist = therapist
        self.history: list[dict] = []

    def _add_to_history(self, role: str, response: str) -> None:
        self.history.append({"role": role, "response": response})

    def out_to_file(self, file_name: str) -> None:
        with open(file_name, "w", encoding="utf-8") as f:
            for message in self.history:
                f.write(f"{message['response']}\n")
        logger.success(f"Output to {file_name}")

    def step(
        self,
        iterative_rewrites,
        inference_object: Agent,
        turn=-1,
        force_ending: str = "",
        repetitive=False,
        repeat_response="",
    ) -> str:
        judge_comment: str = ""
        response: str = ""
        if repeat_response != "" and repetitive:
            judge_comment = f"Your utterance should be different from {repeat_response}"
        if force_ending == "You should end the conversation in this turn":
            judge_comment = force_ending
        elif force_ending != "":
            judge_comment += f"\n{force_ending}"
        for rewrite_turn in range(iterative_rewrites + 1):
            response = inference_object.generate(self.history, comment=judge_comment)
            response = inference_object.post_process(response, turn)
            if rewrite_turn == iterative_rewrites:
                break

            if force_ending == "You should end the conversation in this turn":
                judge_comment = force_ending
            elif force_ending != "" and "[/END]" not in judge_comment:
                judge_comment += f"\n{force_ending}"
            elif force_ending != "" and "[/END]" in response:
                judge_comment = (
                    "You can end the conversation in this turn with '[/END]' token."
                )

            if repetitive:
                judge_comment += f"\nYour utterance should be different from {response}"

        return response

    def run(self, iterative_rewrites: int = 0) -> None:
        start_time = time()
        logger.info("Session started")
        turn = 1
        iterative_rewrites = 1 if random.random() < 0.3 else iterative_rewrites
        therapist_response = self.step(iterative_rewrites, self.therapist, turn)
        self._add_to_history("Therapist", therapist_response)
        logger.info(therapist_response)

        client_response = self.step(iterative_rewrites, self.client, turn)
        self._add_to_history("Client", client_response)
        logger.info(client_response)
        force_ending: str = ""
        while True:
            iterative_rewrites = 1 if random.random() < 0.3 else iterative_rewrites
            therapist_response: str = self.step(
                iterative_rewrites, self.therapist, turn, force_ending=force_ending
            )
            if turn < 15 and "[/END]" in therapist_response:
                therapist_response = therapist_response.replace(" [/END]", "")
            logger.debug(
                "*** Therapist's utterance duplicate: {}",
                {"role": "Therapist", "response": therapist_response} in self.history,
            )
            attempt_count = 0
            max_attempts = 4
            while ({"role": "Therapist",
                    "response": therapist_response,
            } in self.history or not any(therapist_response.startswith(f"{t}:") for t in THERAPIST_TAGS)) and attempt_count < max_attempts:
                logger.warning(
                    f"We are here because the therapist's response is duplicated or not starting with Therapist: , it is: {therapist_response}"
                )
                attempt_count += 1
                therapist_response = self.step(
                    iterative_rewrites,
                    self.therapist,
                    force_ending=force_ending,
                    repetitive=True,
                    repeat_response=therapist_response,
                )

                if attempt_count >= max_attempts:
                    logger.warning(
                        f"Max attempts reached for therapist response, using hardcoded fallback. "
                        f"Questionnaire: {self.therapist.questionnaire_name}, LLM: {self.therapist.llm_type}"
                    )
                    if lang == "fr":
                        therapist_response = random.choice([
                            "Thérapeute: Je vous entends.",
                            "Thérapeute: Pouvez-vous m'en dire plus ?",
                            "Thérapeute: Comment vous sentez-vous en ce moment ?",
                            "Thérapeute: Prenons le temps d'y réfléchir ensemble.",
                            "Thérapeute: Je suis là pour vous écouter.",
                        ])
                    elif lang == "ar":
                        therapist_response = random.choice([
                            "المعالج: أنا أسمعك.",
                            "المعالج: هل يمكنك إخباري بالمزيد؟",
                            "المعالج: كيف تشعر في هذه اللحظة؟",
                            "المعالج: دعنا نفكر في هذا معًا.",
                            "المعالج: أنا هنا للاستماع إليك.",
                        ])
                    elif lang == "de":
                        therapist_response = random.choice([
                            "Therapeut: Ich höre Ihnen zu.",
                            "Therapeut: Können Sie mir mehr darüber erzählen?",
                            "Therapeut: Wie fühlen Sie sich gerade?",
                            "Therapeut: Lassen Sie uns gemeinsam darüber nachdenken.",
                            "Therapeut: Ich bin hier, um Ihnen zuzuhören.",
                        ])
                    break

            self._add_to_history("Therapist", therapist_response)
            logger.info(therapist_response)

            client_response: str = self.step(
                iterative_rewrites, self.client, turn, force_ending=force_ending
            )
            turn += 1
            logger.debug(
                "*** Client's utterance duplicate: {}",
                {"role": "Client", "response": client_response} in self.history,
            )

            # Check if the session should end
            if (
                (
                    self.history[-1] == self.history[-3]
                    and self.history[-2] == self.history[-4]
                )
                or {"role": "Client", "response": client_response} in self.history
                or not any(client_response.startswith(f"{c}:") for c in CLIENT_TAGS)
            ):
                logger.warning(
                    f"We are here because the client's response is duplicated or not starting with Client: , it is: {client_response}"
                )
                cr = client_response.strip()
                if lang == "fr":
                    choice = random.choice([
                    "Client: [Pas de réponse]",
                    "Client: [Pause et réfléchit]",
                    "Client: [Reste silencieux]",
                    "Client: [Silence]",
                    "Client: Je ne sais pas quoi dire",
                    "Client: Je ne sais pas",
                ])
                elif lang == "en":
                    choice = random.choice([
                        "Client: [No reply]",
                        "Client: [Pause and thinking]",
                        "Client: [Keep silent]",
                        "Client: [Quiet]",
                        "Client: I don't know what to say",
                        "Client: I don't know",
                    ])
                elif lang == "de":
                    choice = random.choice([
                        "Klient: [Keine Antwort]",
                        "Klient: [Pause und denkt nach]",
                        "Klient: [Schweigt]",
                        "Klient: [Still]",
                        "Klient: Ich weiß nicht, was ich sagen soll",
                        "Klient: Ich weiß nicht",
                    ])
                elif lang == "ar":
                    choice = random.choice([
                        "العميل: [لا رد]",
                        "العميل: [وقفة قصيرة ويفكر]",
                        "العميل: [يلتزم الصمت]",
                        "العميل: [صمت]",
                        "العميل: لا أعرف ماذا أقول",
                        "العميل: لا أعرف",
                    ])
                self._add_to_history("Client", choice)
                logger.info(choice)
            elif "[/END]" in client_response or "[/END]" in therapist_response:
                logger.success("Conversation takes: {}", turn)
                logger.info(client_response)
                if {
                    "role": "Client",
                    "response": client_response.replace(" [/END]", ""),
                } not in self.history:
                    if "[/END]" in client_response:
                        self._add_to_history("Client", client_response)
                    else:
                        self._add_to_history("Client", client_response + " [/END]")
                else:
                    if "[/END]" not in self.history[-1]["response"]:
                        self.history[-1]["response"] += " [/END]"
                    self._add_to_history("Client", client_response)
                break
            elif turn >= 22:
                logger.warning(
                    "Conversation takes too long, try asking agents to end the conversation"
                )
                logger.info(client_response)
                self._add_to_history("Client", client_response)
                turn_to_end = 30 - turn
                if turn_to_end == 1:
                    force_ending = (
                        f"You should end the conversation in {turn_to_end} turn."
                    )
                elif turn_to_end == 0:
                    force_ending = "You should end the conversation in this turn with '[/END]' token."
                else:
                    force_ending = (
                        f"You should end the conversation in {turn_to_end} turns."
                    )
            else:
                logger.info(client_response)
                self._add_to_history("Client", client_response)
            if turn >= 33:
                logger.error(
                    "Conversation takes too long, force ending the conversation"
                )
                break
            logger.info("Turn: {}", turn)

        logger.info("Session ended, out to file")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        file_path = (
    ""
    f"{self.therapist.questionnaire_name}.txt"
)

        print("Saving to:", file_path)
        print("History size:", len(self.history))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.out_to_file(file_path)
          

        # Calculate elapsed time
        end_time = time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        logger.info(
            "Total time elapsed: {:02d}:{:02d}:{:02d}",
            hours, minutes, seconds
        )
        logger.success("Session ended")