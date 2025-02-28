#!/usr/bin/env python3
import os
import argparse
import time
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
import sys
import threading
import re

# Configure Rich console
console = Console()

# Configuration
RAG_SERVER_URL = "http://localhost:5000"  # RAG server address
CHAT_MODEL = "codellama" ##"deepseek-r1:14b"  # Default chat model other good ones: codegemma:instruct
#Good models I tried:
#   deepseek-r1:14b
#   codegemma:instruct
MAX_RESULTS = 5  # Default number of results to display

class StreamingBuffer:
    """A buffer to hold streaming text and update the display."""

    def __init__(self, query):
        self.text = ""
        self.final_text = ""
        self.query = query
        self.done = False
        self.lock = threading.Lock()
        self.in_thinking = False
        self.thinking_text = ""

    def update(self, chunk):
        """Update the buffer with a new chunk."""
        with self.lock:
            # Check for thinking markers
            if chunk == "<<THINKING_START>>":
                self.in_thinking = True
                return
            elif chunk == "<<THINKING_END>>":
                self.in_thinking = False
                return

            # Add text to appropriate buffer
            if self.in_thinking:
                self.thinking_text += chunk
            else:
                self.final_text += chunk

            # Always update the display text
            self.text += chunk

    def get_panel(self):
        """Get a Rich panel with the current text."""
        with self.lock:
            if self.done:
                # When done, only show the final text (without thinking sections)
                clean_text = re.sub(r'<think>.*?</think>', '', self.final_text, flags=re.DOTALL)
                return Panel(
                    Markdown(clean_text),
                    title=f"[bold green]Answer to: {self.query}[/bold green]",
                    expand=True
                )
            else:
                # During streaming, show thinking sections in a different color
                text_with_highlighted_thinking = self.text
                if "<think>" in text_with_highlighted_thinking:
                    text_with_highlighted_thinking = text_with_highlighted_thinking.replace("<think>", "[bold yellow]💭 THINKING: [/bold yellow]")
                if "</think>" in text_with_highlighted_thinking:
                    text_with_highlighted_thinking = text_with_highlighted_thinking.replace("</think>", "[bold yellow] (End thinking)[/bold yellow]")

                return Panel(
                    Markdown(text_with_highlighted_thinking),
                    title=f"[bold green]Answer to: {self.query}[/bold green]",
                    expand=True
                )

    def mark_done(self):
        """Mark the stream as complete."""
        with self.lock:
            self.done = True
            # When done, make sure to clean up any thinking tags in final output
            self.final_text = re.sub(r'<think>.*?</think>', '', self.final_text, flags=re.DOTALL)

class ManPageQuerier:
    """Class to query the manpage database via rag_server API."""

    def __init__(self, server_url=RAG_SERVER_URL):
        self.server_url = server_url

        # Test the connection
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                console.print("[green]Connected to RAG server successfully[/green]")
            else:
                console.print(f"[red]Failed to connect to RAG server: {response.status_code}[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]Failed to connect to RAG server: {e}[/red]")
            sys.exit(1)

    def query(self, query_text, num_results=MAX_RESULTS, command_filter=None, section_filter=None, diverse_commands=True):
        """Query the database for relevant manpage chunks with command diversity."""
        start_time = time.time()

        # Increase number of results initially fetched to allow for filtering
        fetch_results = num_results * 3 if diverse_commands else num_results

        # Prepare filter parameters
        params = {
            "query": query_text,
            "n_results": fetch_results
        }

        if command_filter:
            params["command"] = command_filter
            diverse_commands = False  # Don't apply diversity if filtering by command
            print(f"DEBUG: Searching for command: '{command_filter}'")
            # Check first few results in the database to see their format
            params={"query": "examples",
                    "n_results": 5}
            if section_filter:
                params["command"] = command_filter + "." + section_filter
            else:
                print(f"DEBUG: section_filter={section_filter}")
                params["command"] = command_filter
            print(f"DEBUG: {params}")
            sample_results = requests.get(f"{self.server_url}/search",
                                          params).json()
            print("DEBUG: Sample commands in database:")
            for r in sample_results.get("results", []):
                print(f"  - {r['command']}")
        if section_filter:
            params["section"] = section_filter

        # Perform the query
        console.print("Searching man pages...")
        try:
            response = requests.get(
                f"{self.server_url}/search",
                params=params
            )

            if response.status_code != 200:
                console.print(f"[red]Error from RAG server: {response.status_code} - {response.text}[/red]")
                return [], 0.0

            result = response.json()

            # Format results from server response
            formatted_results = result.get("results", [])

            # Apply diversity filtering if requested and no command filter is active
            if diverse_commands and len(formatted_results) > num_results:
                filtered_results = []
                seen_commands = set()

                # First pass: select one entry from each unique command
                for result in formatted_results:
                    command = result["command"]
                    if command not in seen_commands:
                        seen_commands.add(command)
                        filtered_results.append(result)

                        # If we have enough diverse commands, stop
                        if len(filtered_results) >= num_results:
                            break

                # Second pass: if we need more results, add additional relevant chunks
                if len(filtered_results) < num_results:
                    for result in formatted_results:
                        if result not in filtered_results and len(filtered_results) < num_results:
                            filtered_results.append(result)

                formatted_results = filtered_results

            # Ensure we don't return more than requested
            formatted_results = formatted_results[:num_results]

            elapsed_time = time.time() - start_time

            return formatted_results, elapsed_time

        except Exception as e:
            console.print(f"[red]Error querying RAG server: {e}[/red]")
            return [], 0.0

    def generate_answer(self, query_text, context_chunks, model=CHAT_MODEL, stream=False):
        """Generate an answer based on retrieved chunks using the RAG server."""
        # Count unique commands
        unique_commands = {c["command"] for c in context_chunks}
        has_multiple_commands = len(unique_commands) > 1

        # Create contexts grouped by command
        context_by_command = {}
        for chunk in context_chunks:
            cmd = chunk["command"]
            if cmd not in context_by_command:
                context_by_command[cmd] = []
            context_by_command[cmd].append(chunk)

        # Format multi-command context
        formatted_chunks = []
        for chunk in context_chunks:
            formatted_chunk = chunk.copy()
            command_name = chunk["command"]
            if "." in command_name and command_name.split(".")[-1].isdigit():
                formatted_chunk["display_command"] = command_name.rsplit(".", 1)[0]
            else:
                formatted_chunk["display_command"] = command_name
            formatted_chunks.append(formatted_chunk)

        # Prepare request body with special flags for multi-command context
        request_body = {
            "query": query_text,
            "contexts": formatted_chunks,
            "model": model,
            "stream": stream
        }

        if has_multiple_commands:
            request_body["compare_commands"] = True
            request_body["commands_list"] = list(unique_commands)

        if not stream:
            console.print("Generating answer...")
            try:
                response = requests.post(
                    f"{self.server_url}/generate",
                    json={
                        "query": query_text,
                        "contexts": context_chunks,
                        "model": model,
                        "stream": False
                    }
                )

                if response.status_code != 200:
                    console.print(f"[red]Error generating answer: {response.status_code} - {response.text}[/red]")
                    return None

                result = response.json()
                if "answer" not in result:
                    console.print(f"[red]No answer found in response[/red]")
                    return None

                return result["answer"]
            except Exception as e:
                console.print(f"[red]Error generating answer: {e}[/red]")
                return None
        else:
            # For streaming, we return a buffer object and update it in a separate thread
            buffer = StreamingBuffer(query_text)

            def stream_answer():
                try:
                    with requests.post(
                        f"{self.server_url}/generate_stream",
                        json={
                            "query": query_text,
                            "contexts": context_chunks,
                            "model": model,
                            "stream": True
                        },
                        stream=True
                    ) as response:
                        if response.status_code != 200:
                            console.print(f"[red]Error streaming answer: {response.status_code} - {response.text}[/red]")
                            buffer.mark_done()
                            return

                        # Process each chunk as it arrives
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                buffer.update(chunk)

                        buffer.mark_done()
                except Exception as e:
                    console.print(f"[red]Error streaming answer: {e}[/red]")
                    buffer.mark_done()

            # Start streaming in a background thread
            threading.Thread(target=stream_answer).start()
            return buffer

def display_results(results, elapsed_time, query):
    """Display search results in a nice format."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(results)} results in {elapsed_time:.4f} seconds[/bold green]")
    console.print(f"[bold]Query:[/bold] {query}\n")

    # Count unique commands
    unique_commands = {r["command"] for r in results}
    console.print(f"[bold cyan]Found information about {len(unique_commands)} different commands:[/bold cyan] " +
                  ", ".join(sorted(unique_commands)))

    # Group by command
    command_groups = {}
    for result in results:
        cmd = result["command"]
        if cmd not in command_groups:
            command_groups[cmd] = []
        command_groups[cmd].append(result)

    # Create a table for results with command grouping
    for cmd, cmd_results in command_groups.items():
        table = Table(title=f"Man Page Results for '{cmd}'", expand=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Sec", style="magenta", no_wrap=True)
        table.add_column("Relevance", style="green", no_wrap=True)
        table.add_column("Content Preview", style="white")

        for result in cmd_results:
            # Truncate content for preview
            content_preview = result["content"]
            if len(content_preview) > 100:
                content_preview = content_preview[:97] + "..."

            # Add row to table
            command_name = result["command"]
            # Remove section number if it's in the command name (like "openvt.1")
            if "." in command_name and command_name.split(".")[-1].isdigit():
                command_name = command_name.rsplit(".", 1)[0]
            table.add_row(
                result["command"],
                result["section"],
                f"{result.get('relevance', 0.0):.4f}",
                content_preview
            )

        console.print(table)
        console.print()

def display_detailed_result(result):
    """Display a single result in detail."""
    command_name = result["command"]
    if "." in command_name and command_name.split(".")[-1].isdigit():
        command_name = command_name.rsplit(".", 1)[0]
    panel = Panel(
        result["content"],
        title=f"[bold cyan]{command_name}({result['section']})[/bold cyan]",
        subtitle=f"Relevance: {result.get('relevance', 0.0):.4f}",
        expand=True
    )
    console.print(panel)
    console.print()

def display_answer(answer, query):
    """Display the generated answer."""
    if isinstance(answer, StreamingBuffer):
        # For streaming answers
        with Live(answer.get_panel(), refresh_per_second=4) as live:
            while not answer.done:
                live.update(answer.get_panel())
                time.sleep(0.25)
            # Final update
            live.update(answer.get_panel())
    elif answer:
        # For non-streaming answers
        panel = Panel(
            Markdown(answer),
            title=f"[bold green]Answer to: {query}[/bold green]",
            expand=True
        )
        console.print(panel)
    else:
        console.print("[red]Failed to generate an answer.[/red]")

def display_command_list(results):
    """Display a list of unique commands with brief descriptions."""
    # Extract unique commands
    commands = {}
    for result in results:
        cmd = result["command"]
        if "." in cmd and cmd.split(".")[-1].isdigit():
            cmd = cmd.rsplit(".", 1)[0]

        if cmd not in commands:
            # Get a brief description (first sentence or first 100 chars)
            content = result["content"]
            desc = content.split(". ")[0]
            if len(desc) > 100:
                desc = desc[:97] + "..."
            commands[cmd] = {
                "section": result["section"],
                "description": desc
            }

    # Display command list
    if not commands:
        console.print("[yellow]No commands found.[/yellow]")
        return None

    console.print(f"\n[bold green]Found {len(commands)} relevant commands:[/bold green]\n")

    # Create a table for command list
    table = Table(title="Available Commands", expand=True)
    table.add_column("#", style="bold", no_wrap=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Section", style="magenta", no_wrap=True)
    table.add_column("Brief Description", style="white")

    for i, (cmd, info) in enumerate(commands.items(), 1):
        table.add_row(
            str(i),
            cmd,
            info["section"],
            info["description"]
        )

    console.print(table)
    return list(commands.keys())

def two_step_query(querier, query_text, num_results=MAX_RESULTS,
                   command_filter=None, section_filter=None):
    """Perform a two-step query: first list commands, then get details for selected command."""
    # Step 1: Get diverse results to identify relevant commands
    console.print(f"[bold]Searching for commands {'in section ' + section_filter + ' ' if section_filter else ''}related to:[/bold] {query_text}")
    results, elapsed_time = querier.query(
        query_text=query_text,
        num_results=num_results * 2,  # Get more results to find more commands
        diverse_commands=True,
        command_filter=command_filter,
        section_filter=section_filter
    )

    # Extract and display command list
    commands = display_command_list(results)
    if not commands:
        return

    # Step 2: Let user select a command
    while True:
        selection = console.input("\n[bold green]Select a command[/bold green] (number or name, or 'q' to quit): ")

        if selection.lower() in ('q', 'quit', 'exit'):
            return

        selected_command = None
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(commands):
                selected_command = commands[idx]
            else:
                console.print("[red]Invalid number. Please try again.[/red]")
                continue
        else:
            # Check if input matches a command name
            matches = [cmd for cmd in commands if cmd.lower() == selection.lower()]
            if matches:
                selected_command = matches[0]
            else:
                console.print("[red]Command not found. Please try again.[/red]")
                continue

        # Step 3: Get detailed results for the selected command
        console.print(f"\n[bold]Getting detailed information for:[/bold] {selected_command}")
        # Add section number to command
        for result in results:
            if result["command"].split(".")[0] == selected_command:
                selected_command = result["command"]
        #selected_command = selected_command + results["command"][
        detailed_results, _ = querier.query(
            query_text=query_text,
            num_results=num_results,
            command_filter=selected_command,
            diverse_commands=False
        )

        # Display detailed results
        display_results(detailed_results, elapsed_time, query_text)

        # Optionally show full content
        if detailed_results and console.input("\nShow detailed content? [y/N]: ").lower() == 'y':
            for result in detailed_results:
                display_detailed_result(result)

        # Generate answer if requested
        if detailed_results:
            stream = console.input("\nGenerate streaming answer? [y/N]: ").lower() == 'y'
            if stream or console.input("\nGenerate answer? [y/N]: ").lower() == 'y':
                answer = querier.generate_answer(query_text, detailed_results, stream=stream)
                display_answer(answer, query_text)

        # Ask if user wants to select another command
        if console.input("\nSelect another command? [y/N]: ").lower() != 'y':
            break

def main():
    parser = argparse.ArgumentParser(description="Query the man page RAG database")
    parser.add_argument("query", nargs="?", help="The query to search for")
    parser.add_argument("-c", "--command", help="Filter results by command name")
    parser.add_argument("-s", "--section", help="Filter results by man section number")
    parser.add_argument("-n", "--num-results", type=int, default=MAX_RESULTS, help="Number of results to display")
    parser.add_argument("-d", "--detail", action="store_true", help="Show detailed results")
    parser.add_argument("-a", "--answer", action="store_true", help="Generate an answer from the results")
    parser.add_argument("-m", "--model", default=CHAT_MODEL, help="Model to use for answer generation")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--server", default=RAG_SERVER_URL, help="RAG server URL")
    parser.add_argument("--stream", action="store_true", help="Stream the answer as it's generated", default=True)
    parser.add_argument("--diverse", action="store_true", default=True, help="Get diverse results across different commands")
    parser.add_argument("--no-diverse", dest="diverse", action="store_false", help="Don't apply command diversity")
    parser.add_argument("--two-step", action="store_true", help="Two-step query: first list commands, then show details")

    args = parser.parse_args()

    # Initialize the querier
    querier = ManPageQuerier(server_url=args.server)

    # Interactive mode
    if args.interactive:
        console.print("[bold cyan]ManPage RAG Query Tool - Interactive Mode[/bold cyan]")
        console.print("Type your queries below. Type 'exit' to quit.\n")

        while True:
            query = console.input("[bold green]Query[/bold green]: ")
            if query.lower() in ('exit', 'quit'):
                break

            command = console.input("[bold cyan]Command filter[/bold cyan] (optional): ")
            if not command:
                command = None

            section = console.input("[bold magenta]Section filter[/bold magenta] (optional): ")
            if not section:
                section = None

            results, elapsed_time = querier.query(
                query_text=query,
                num_results=args.num_results,
                command_filter=command,
                section_filter=section,
                diverse_commands=args.diverse
            )

            display_results(results, elapsed_time, query)

            if results and console.input("\nShow details? [y/N]: ").lower() == 'y':
                for result in results:
                    display_detailed_result(result)

            if results:
                stream = console.input("\nGenerate streaming answer? [y/N]: ").lower() == 'y'
                if stream or console.input("\nGenerate answer? [y/N]: ").lower() == 'y':
                    answer = querier.generate_answer(query, results, model=args.model, stream=stream)
                    display_answer(answer, query)

            console.print("\n" + "-" * 50 + "\n")
    # Two-step query
    elif args.two_step and args.query:
        print(f"DEBUG: args.command={args.command}")
        print(f"DEBUG: args.section={args.section}")
        two_step_query(querier, args.query, args.num_results,
                       command_filter=args.command, section_filter=args.section)
    # Single query mode
    elif args.query:
        results, elapsed_time = querier.query(
            query_text=args.query,
            num_results=args.num_results,
            command_filter=args.command,
            section_filter=args.section,
            diverse_commands=args.diverse
        )

        display_results(results, elapsed_time, args.query)

        if results and args.detail:
            for result in results:
                display_detailed_result(result)

        if results and args.answer:
            answer = querier.generate_answer(args.query, results, model=args.model, stream=args.stream)
            display_answer(answer, args.query)
    else:
        console.print("[yellow]Please provide a query or use --interactive mode.[/yellow]")
        parser.print_help()

if __name__ == "__main__":
    main()
