import os
import typer
import openai


# Prompt Template

# Use [TARGETLANGUAGE] and [PROMPT] to write a [DIAGRAM TYPE] diagram for [PURPOSE] with [DIAGRAMMING TOOL] script.
# Your diagram should clearly depict [NUMBER] [ELEMENT TYPE] and should be optimized for easy understanding.

# Teaser
# Improve your diagramming skills with [DIAGRAMMING TOOL] by creating optimized diagrams for various purposes using [DIAGRAM TYPE].

# Prompt Arguments
# [DIAGRAM TYPE] - Sequence, Use Case, Class, Activity, Component, State, Object, Deployment, Timing, Network, Wireframe, Archimate, Gantt, MindMap, WBS, JSON, YAML 
# [ELEMENT TYPE] - Actors, Messages, Objects, Classes, Interfaces, Components, States, Nodes, Edges, Links, Frames, Constraints, Entities, Relationships, Tasks, Events, Modules 
# [PURPOSE] - Communication, Planning, Design, Analysis, Modeling, Documentation, Implementation, Testing, Debugging
# [DIAGRAMMING TOOL] - PlantUML, Mermaid, Draw.io, Lucidchart, Creately, Gliffy, Structurizr DSL

# Title: Diagramming UML
# Topic: Software Applications
# Activity: Design

# Example:
# Use English and [DIAGRAM TYPE] - Sequence [ELEMENT TYPE] -  Messages [PURPOSE] - Communication frontend backend [DIAGRAMMING TOOL] - PlantUML to
# write a [DIAGRAM TYPE] diagram for [PURPOSE] with [DIAGRAMMING TOOL] script. Your diagram should clearly depict [NUMBER] [ELEMENT TYPE] and should be optimized for easy understanding.

openai.api_key = ""
model_engine = "gpt-3.5-turbo"

app = typer.Typer()

@app.command(
    help="Generates a UML script for any platform from a text description and saves it to the specified path."
)
def generate_chatgpt_prompt(
    target_language: str = typer.Argument(...),
    prompt: str = typer.Argument(...),
    diagram_type: str = typer.Argument(...),
    purpose: str = typer.Argument(...),
    diagramming_tool: str = typer.Argument(...),
    number: str = typer.Argument(...),
    element_type: str = typer.Argument(...),
    title: str = typer.Argument(...),
    topic: str = typer.Argument(...),
):
    """
    Generates a UML script for any platform from a text description and saves it to the specified path.
    :param text: The text description
    :param output_path: The path where the generated UML script will be saved
    """

    # Prompt Template

    messages = [
        {
            "role": "system",
            "content": f"Use {target_language} to write a {diagram_type} diagram for {purpose} with {diagramming_tool} script.",
        },
        {
            "role": "system",
            "content": f"Your diagram should clearly depict {number} {element_type} and should be optimized for easy understanding.",
        },
        {
            "role": "system",
            "content": f"Improve your diagramming skills with {diagramming_tool} by creating optimized diagrams for various purposes using {diagram_type}.",
        },
        {
            "role": "system",
            "content": "[DIAGRAM TYPE] - Sequence, Use Case, Class, Activity, Component, State, Object, Deployment, Timing, Network, Wireframe, Archimate, Gantt, MindMap, WBS, JSON, YAML",
        },
        {
            "role": "system",
            "content": "[ELEMENT TYPE] - Actors, Messages, Objects, Classes, Interfaces, Components, States, Nodes, Edges, Links, Frames, Constraints, Entities, Relationships, Tasks, Events, Modules",
        },
        {
            "role": "system",
            "content": "[PURPOSE] - Communication, Planning, Design, Analysis, Modeling, Documentation, Implementation, Testing, Debugging",
        },
        {
            "role": "system",
            "content": "[DIAGRAMMING TOOL] - PlantUML, Mermaid, Draw.io, Lucidchart, Creately, Gliffy, Structurizr DSL",
        },
        {"role": "system", "content": f"Title: {title}"},
        {"role": "system", "content": f"Topic: {topic}"},
        {"role": "system", "content": f"Activity: {purpose}"},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=4096
    )

    typer.echo(str(response.choices))

    return str(response.choices[0].message.content.strip())


@app.command(
    help="Generates a UML script from a text description and saves it to the specified path."
)
def write_script(
    diagram_script_path: str = typer.Argument(...),
    output_path: str = typer.Argument(...),
    write_to_file: bool = typer.Option(False, "--write-to-file", "-w"),
):
    """
    Generates a UML script from a text description and saves it to the specified path.
    :param text: The text description
    :param output_path: The path where the generated UML script will be saved
    """
    # Generate the UML script from the text description
    UML_script = generate_chatgpt_prompt(diagram_script_path)

    if write_to_file:
        with open(output_path, "w") as f:
            f.write(UML_script)

    typer.echo(UML_script)
    typer.echo(f"UML script saved to {output_path}")
    return UML_script, output_path

if __name__ == "__main__":
    # app(
    #     generate_chatgpt_prompt,
    #     target_language="English",
    #     prompt="PlantUML - Sequence Messages Communication frontend backend",
    #     diagram_type="Sequence",
    #     purpose="Communication",
    #     diagramming_tool="PlantUML",
    #     number="2",
    #     element_type="Messages",
    #     title="Diagramming UML",
    #     topic="Software Applications",
    #     activity="Design",
    #     help="Generates a UML script for any platform from a text description and saves it to the specified path.",
    # )

    app()