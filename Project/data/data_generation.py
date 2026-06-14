import json
import random

# --- Configuration & Pools ---
PRINCIPAL_POOL = ["User", "Manager", "Admin", "Guest", "System", "Auditor", "Developer"]
ACTION_POOL = ["write", "create", "delete", "edit", "approve", "view", "reject"]
RESOURCE_POOL = ["Invoice", "Ticket", "Order", "Presentation", "Report", "Document"]
APP_NAMES = ["InternalApp", "SecureData", "ProjectHub", "AuthGuard"]

def get_relational_logic(principal_type):
    """Returns logic specifically tied to one principal type to ensure strict ownership."""
    choices = [
        {"desc": f"owned by the {principal_type}", "attr": "owner", "logic": "resource has owner && resource.owner == principal", "type": "Entity"},
        {"desc": f"assigned to the {principal_type}", "attr": "assignee", "logic": "resource has assignee && resource.assignee == principal", "type": "Entity"},
        {"desc": "within their project", "attr": "project", "logic": "resource has project && principal has project && resource.project == principal.project", "type": "String"},
    ]
    return random.choice(choices)

def generate_unstructured_prompt(app, allows, forbids):
    reqs = allows + [f"STRICTLY FORBID: {f}" for f in forbids]
    random.shuffle(reqs)
    styles = [
        f"Generate Cedar code for {app}. Rules: " + " | ".join(reqs),
        f"In {app}, we need these permissions: " + ", ".join(allows) + ". Don't allow: " + " or ".join(forbids),
        f"Draft a Cedar schema and policy set for {app}. Requirements:\n" + "\n".join([f"- {r}" for r in reqs]),
        f"How do I write Cedar for {app}? {random.choice(allows)}. Also block if {random.choice(forbids) if forbids else 'it is private'}.",
    ]
    return random.choice(styles)

def generate_datasets(num_samples=1000):
    with open("cedar_unstructured_dataset.jsonl", "w") as f_unstruct, \
         open("cedar_structured_dataset.jsonl", "w") as f_struct:
        
        for _ in range(num_samples):
            app = random.choice(APP_NAMES)
            # Pick exactly one primary principal and one secondary to keep ownership clear
            principals = list(set(random.choices(PRINCIPAL_POOL, k=2)))
            resources = list(set(random.choices(RESOURCE_POOL, k=random.randint(2, 3))))
            actions = list(set(random.choices(ACTION_POOL, k=4)))
            
            allow_texts, forbid_texts = [], []
            policies_cedar = []
            res_attrs = {res: {} for res in resources}
            pri_attrs = {pri: {} for pri in principals}

            # To ensure ONE owner per resource, we map each resource to one specific principal type for the whole sample
            resource_owner_map = {res: random.choice(principals) for res in resources}

            # Generate 2-3 Permit Policies
            for _ in range(random.randint(2, 3)):
                r = random.choice(resources)
                p = resource_owner_map[r] # Force the principal to be the designated type for this resource
                a = random.choice(actions)
                rel = get_relational_logic(p)
                
                allow_texts.append(f"{p} is allowed to {a} {r} {rel['desc']}")
                policies_cedar.append(
                    f"- POLICY_{len(policies_cedar)+1}\n  permit (\n    principal is {app}::{p},\n    action == {app}::Action::\"{a}\",\n    resource is {app}::{r}\n  ) when {{\n    {rel['logic']}\n  }};"
                )
                
                if rel['type'] == "Entity":
                    res_attrs[r][rel['attr']] = {"type": "Entity", "name": p, "required": True}
                elif rel['type'] == "String":
                    res_attrs[r]["project"] = {"type": "String", "required": True}

            # Generate Forbid Logic (if applicable)
            if random.random() < 0.6:
                r = random.choice(resources)
                p = random.choice(principals)
                a = random.choice(actions)
                forbid_texts.append(f"{p} cannot {a} {r} if it is not belonging to their project")
                policies_cedar.append(
                    f"- POLICY_{len(policies_cedar)+1}\n  forbid (\n    principal is {app}::{p},\n    action == {app}::Action::\"{a}\",\n    resource is {app}::{r}\n  ) when {{\n    resource has project && principal has project && resource.project != principal.project\n  }};"
                )
                res_attrs[r]["project"] = {"type": "String", "required": True}

            # Build Schema
            schema = {app: {"entityTypes": {}, "actions": {}}}
            for p in principals: schema[app]["entityTypes"][p] = {"shape": {"type": "Record", "attributes": pri_attrs[p]}}
            for r in resources: schema[app]["entityTypes"][r] = {"shape": {"type": "Record", "attributes": res_attrs[r]}}
            for a in actions: schema[app]["actions"][a] = {"appliesTo": {"principalTypes": principals, "resourceTypes": resources}}

            final_output = f"Schema:\n{json.dumps(schema, indent=2)}\n\nPolicies:\n" + "\n".join(policies_cedar)
            
            # Write files
            f_unstruct.write(json.dumps({"input": generate_unstructured_prompt(app, allow_texts, forbid_texts), "output": final_output}) + "\n")
            f_struct.write(json.dumps({"input": json.dumps({"app": app, "principals": principals, "actions": actions, "resources": resources, "permissions": allow_texts, "forbidden_permissions": forbid_texts}), "output": final_output}) + "\n")

    print(f"Dataset generated with Single-Owner enforcement.")

if __name__ == "__main__":
    generate_datasets(1000)