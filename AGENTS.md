# Agent Instructions

Use the principles outlined below to guide your planning and activities.

## Test-Driven Development (TDD)
- Always write a failing test before implementing any new functionality or fixing a new bug.
- Ensure the test suite is capable of running and detecting the failure.
- Use black box testing and development concepts
- Once a block of functionality is completed and green, test it's entire functionality again.

## Red-Green-Refactor Cycle
1. **Red**: Write a test for the next bit of functionality you want to add. Run the test and see it fail.
2. **Green**: Write the minimum amount of code necessary to make the test pass.
3. **Refactor**: Clean up the code you just wrote, ensuring it adheres to project standards and patterns while keeping the tests passing.

## SOLID principles
- Single responsibility principle - each function must perform one specific task; each test must test one specific failure case
- Open/Closed Principle (OCP) - You should be able to add new functionality to a class without changing its existing source code. This is usually achieved using interfaces or abstract classes.
- Liskov Substitution Principle: A derived (child) class must be completely substitutable for its base (parent) class. If a subclass cannot perform the same actions as its parent, it violates this principle. You should be able to add new functionality to a class without changing its existing source code.
- It is better to have many small, specific interfaces than one large, "fat" interface. Classes should only have to implement the methods they actually need.
- High-level modules (the "logic" of your app) should not depend on low-level modules (the "tools" like databases or APIs). Both should depend on abstractions (interfaces).

## Git Checkpoints
- Create a git commit for every green test.
- Use descriptive commit messages that reflect the current state (e.g., `test: add failing test for ...`, `feat: implement ...`, `refactor: clean up ...`).
- DO NOT git push without explicit instructions.
- Do not git commit when tests are red.

## Requirements and Instructions
- Assume as little as possible - clarify any requirements that are unclear
- Do not use emojis in code and comments
