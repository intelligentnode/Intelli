"""
Live VibeAgent tests on the IntelliNode use cases: each test plans a request with a real planner, checks the plan,
then runs the flow on sample input and checks the output.

    OPENAI_API_KEY              the planner (VIBE_PLANNER_MODEL, default gpt-5.5) and most steps
    GEMINI_API_KEY              web facts and the news digest (Google Search grounding)
    MISTRAL_API_KEY             the travel guide
    QDRANT_URL                  answers from a Qdrant collection (docker run -d -p 6333:6333 qdrant/qdrant)
    VIBE_MEDIA_TESTS=1          the cases that generate images and audio (billed per call)
    VIBE_COMPUTER_TESTS=1       the computer-use cases on a local staging site: ANTHROPIC_API_KEY,
                                pip install "intelli[computer]" and playwright install chromium

Run:
    python3 -m pytest intelli/test/integration/test_vibe_use_cases_live.py -q -s
"""
import asyncio
import json
import os
import shutil
import tempfile
import unittest

import requests
from dotenv import load_dotenv

from intelli.flow.vibe import VibeAgent
from intelli.test.integration import vibe_staging_site as site

load_dotenv()

LANGUAGE = {'assistant', 'text'}
HANDBOOK = """# Acme customer handbook

## Returns
Customers can return items within 30 days of delivery for a full refund. Laptops are an exception: they can be
returned within 15 days. Opened software cannot be returned.

## Shipping
Standard shipping takes 5 business days and is free over 50 dollars. Express shipping takes 2 business days and
costs 9 dollars.
"""
HR_POLICY = """# Acme HR policy

## Vacation
Full-time employees get 25 vacation days a year. Unused days carry over until March 31 of the next year.

## Remote work
Employees can work remotely up to three days a week.
"""
CATALOG = """sku,name,price,stock,notes
LS-01,Aluminum laptop stand,49.00,12,fits 10-17 inch laptops
LS-02,Adjustable laptop stand with fan,64.50,0,back in stock in two weeks
LS-03,Foldable laptop stand,29.99,40,weighs 300 grams
"""
COMMITS = """feat: sign in with Google Workspace accounts
feat: export invoices as CSV from the billing page
fix: app crashed when uploading a PDF larger than 10 MB
fix: receipts showed the wrong currency symbol for EUR
feat: dark mode in the settings page
chore: upgrade the payments SDK to v5 (changes retry behavior)"""
BLOG_POST = """Why we moved our office to a four-day week

Last spring our 40-person team tried a four-day week for three months. We kept the same salaries and the same
goals. Support response times stayed under two hours, and we shipped 12 releases instead of the usual 10. Sick
days dropped by a third. We cut recurring meetings by half and moved status updates to a written channel. After
the trial, 37 of the 40 people voted to keep the four-day week, and we made it permanent in September."""
DIFF = """--- a/app/users.py
+++ b/app/users.py
@@ def find_user(name):
-    return db.execute("SELECT * FROM users WHERE name = ?", (name,))
+    return db.execute(f"SELECT * FROM users WHERE name = '{name}'")
"""


def get_weather(city: str) -> dict:
    """Current weather of a city: forecast and temperature in Celsius."""
    return {'city': city, 'forecast': 'light rain', 'temperature_c': 14}


GUARD_CALLS = []


def read_only(action):
    """Blocks actions that would place an order, pay, submit, approve or delete."""
    GUARD_CALLS.append(action)
    text = f"{action.get('text', '')} {action.get('target_text', '')}".lower()
    return not any(word in text for word in ('place order', 'pay', 'submit', 'approve', 'delete', 'confirm'))


@unittest.skipUnless(os.getenv('OPENAI_API_KEY'), 'Set OPENAI_API_KEY')
class TestVibeUseCasesLive(unittest.TestCase):
    """Each test plans one request; the plan is printed with -s."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.folder = tempfile.mkdtemp()
        os.chdir(self.folder)  # specs use relative paths such as ./handbook.md and ./conversations
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.addCleanup(os.chdir, self.cwd)
        self.spec = None

    # ------------------------------------------------------------------ helpers
    def vibe(self, **registries):
        return VibeAgent(planner_provider='openai', planner_api_key=os.environ['OPENAI_API_KEY'],
                         planner_model=os.getenv('VIBE_PLANNER_MODEL', 'gpt-5.5'), **registries)

    def build(self, vibe, request):
        flow = asyncio.run(vibe.build(request, save_dir=self.folder))
        self.spec = vibe.last_spec
        print(f'\n[{self._testMethodName}] plan:')
        for task in self.spec['tasks']:
            agent = task['agent']
            params = {k: v for k, v in (agent.get('model_params') or {}).items() if k != 'key'}
            print(f"  - {task['name']} [{agent['agent_type']}:{agent.get('provider')}] {params} "
                  f"options={json.dumps(agent.get('options') or {})[:200]}")
        print(f"  map_paths={self.spec.get('map_paths')} connectors={self.spec.get('dynamic_connectors')}")
        return flow

    def run_flow(self, flow, initial_input=None):
        output = asyncio.run(flow.start(initial_input=initial_input))
        for name, item in output.items():
            value = item['output']
            shown = f'<{item["type"]}>' if item['type'] in ('image', 'audio') else str(value)[:300]
            print(f'  > {name}: {shown}')
        self.assertEqual(flow.errors, {})
        return output

    def language_tasks(self):
        return [t for t in self.spec['tasks'] if t['agent']['agent_type'] in LANGUAGE]

    def tasks(self, agent_type):
        return [t for t in self.spec['tasks'] if t['agent']['agent_type'] == agent_type]

    def assert_assistants(self):
        language = self.language_tasks()
        self.assertTrue(language)
        self.assertTrue(all(t['agent']['agent_type'] == 'assistant' for t in language))

    @staticmethod
    def options(task):
        return task['agent'].get('options') or {}

    @staticmethod
    def params(task):
        return task['agent'].get('model_params') or {}

    def predecessors(self, name):
        return [src for src, dsts in (self.spec.get('map_paths') or {}).items() if name in dsts]

    @staticmethod
    def text(output):
        return ' '.join(str(item['output']).lower() for item in output.values() if item['type'] == 'text')

    # ------------------------------------------------------------------ the article: build AI agents with coding agents
    def test_support_triage(self):
        flow = self.build(self.vibe(), (
            'Build a support ticket triage tool. For each ticket, run three steps in parallel: pick a category '
            '(billing, bug, account, feature), rate urgency (high, medium, low) and write a one line summary. High '
            'urgency tickets get an escalation note for the on-call engineer; the rest get a drafted customer reply.'))
        self.assert_assistants()
        self.assertTrue(self.spec.get('dynamic_connectors'))
        routed = [t for c in self.spec['dynamic_connectors'] for t in c['destinations'].values()]
        escalation = [t for t in routed if 'escal' in t]
        reply = [t for t in routed if t not in escalation]

        high = self.run_flow(flow, 'Since 9am none of our 300 employees can log in to the dashboard. We get a 500 '
                                   'error and the sales team is blocked.')
        low = self.run_flow(flow, 'It would be lovely to have a dark mode option some day. Thanks!')

        self.assertTrue(set(escalation) & set(high) and not set(reply) & set(high))
        self.assertTrue(set(reply) & set(low) and not set(escalation) & set(low))

    def test_release_brief(self):
        flow = self.build(self.vibe(), (
            "Every Friday I paste the week's commit messages. Turn them into a short release brief for customers and "
            'the sales team: a features section, a fixes section and a risks section, written in parallel, then one '
            'brief with a headline.'))
        self.assert_assistants()
        joins = [t['name'] for t in self.spec['tasks'] if len(self.predecessors(t['name'])) >= 3]
        self.assertTrue(joins, 'three parallel writers join in one brief')

        brief = str(self.run_flow(flow, COMMITS)[joins[0]]['output']).lower()

        self.assertTrue(all(word in brief for word in ('feature', 'fix', 'risk')))
        self.assertIn('csv', brief)

    def test_blog_post_to_four_channels(self):
        flow = self.build(self.vibe(), (
            'Turn any blog post into a tweet thread, a LinkedIn post, a newsletter blurb and a search snippet, all '
            'written at the same time. Add a first step that reads the post and picks its key points, and have the '
            'four writers work from those points.'))
        self.assert_assistants()
        fan_out = [src for src, dsts in (self.spec.get('map_paths') or {}).items() if len(dsts) >= 4]
        self.assertTrue(fan_out, 'one step feeds four writers')

        output = self.run_flow(flow, BLOG_POST)

        self.assertEqual(len(output), 5, 'the key points stay in the result')
        self.assertIn('four-day', self.text(output).replace('four day', 'four-day'))

    # ------------------------------------------------------------------ assistants: documents, history, memory, tools
    def test_support_chat_with_handbook_and_history(self):
        with open('handbook.md', 'w') as file:
            file.write(HANDBOOK)
        flow = self.build(self.vibe(), (
            'A support chat for Acme customers. It answers from our handbook in ./handbook.md, shows the sources, and '
            'remembers the conversation so follow-up questions work.'))
        self.assert_assistants()
        answer = [t for t in self.language_tasks() if self.options(t).get('knowledge')]
        self.assertTrue(answer)
        self.assertIn('handbook.md', json.dumps(self.options(answer[0])))
        self.assertTrue(any(self.options(t).get('history') and self.params(t).get('conversation_id')
                            for t in self.language_tasks()))

        first = self.run_flow(flow, 'How long do I have to return a laptop?')
        second = self.run_flow(flow, 'And other items?')

        self.assertIn('15', self.text(first))
        self.assertIn('30', self.text(second))

    def test_cooking_assistant_remembers_a_user(self):
        flow = self.build(self.vibe(), (
            "A cooking assistant that suggests dinners and remembers each user's diet and allergies across "
            'conversations, so they never have to repeat them.'))
        self.assert_assistants()
        remembering = [t['name'] for t in self.language_tasks()
                       if self.options(t).get('memory') and self.params(t).get('user_id')]
        self.assertTrue(remembering)

        self.run_flow(flow, 'I am vegetarian and allergic to peanuts.')
        for name in remembering:  # a new conversation: only long-term memory brings the diet back
            task = flow.tasks[name]
            task.model_params = {**task.model_params, 'conversation_id': 'another-chat'}
        second = self.run_flow(flow, 'What should I cook tonight? One idea only.')

        recalled = [memory['text'] for name in remembering
                    for memory in flow.tasks[name].agent._get_handler().last_reply['memories']]
        self.assertTrue(any('vegetarian' in memory.lower() for memory in recalled))
        self.assertTrue('vegetarian' in self.text(second) or 'peanut' in self.text(second))

    def test_weather_tool(self):
        flow = self.build(self.vibe(tools={'get_weather': get_weather}), (
            'A travel helper that tells the current weather for the city the user asks about and suggests what to '
            'wear.'))
        self.assert_assistants()
        self.assertTrue(any(self.options(t).get('tools') == ['get_weather'] for t in self.language_tasks()))

        output = self.run_flow(flow, 'I land in Oslo tonight. What should I wear?')

        self.assertTrue('14' in self.text(output) or 'rain' in self.text(output))

    def test_product_questions_from_a_catalog(self):
        with open('catalog.csv', 'w') as file:
            file.write(CATALOG)
        flow = self.build(self.vibe(), (
            'A shopping assistant for our store. It answers product questions from our catalog in ./catalog.csv and '
            'remembers what each shopper told us they need, across visits.'))
        self.assert_assistants()
        self.assertTrue(any('catalog.csv' in json.dumps(self.options(t)) for t in self.language_tasks()))
        self.assertTrue(any(self.options(t).get('memory') for t in self.language_tasks()))

        self.run_flow(flow, 'I need a stand for my 16 inch laptop.')
        second = self.text(self.run_flow(flow, 'Is the one with a fan in stock?'))

        self.assertTrue('not' in second or 'out of stock' in second or 'two weeks' in second)

    def test_code_review_routes_security_problems(self):
        flow = self.build(self.vibe(), (
            'Review the code diff in the input. Write review comments for the author. If the diff has a security '
            'problem, also write a short security alert for the security team; otherwise skip the alert.'))
        self.assert_assistants()
        self.assertTrue(self.spec.get('dynamic_connectors'))

        risky = self.run_flow(flow, DIFF)
        safe = self.run_flow(flow, DIFF.replace("db.execute(f\"SELECT * FROM users WHERE name = '{name}'\")",
                                                 'db.execute("SELECT * FROM users WHERE name = ? LIMIT 1", (name,))'))

        alert = [name for name in risky if 'security' in name or 'alert' in name]
        self.assertTrue(alert, 'the SQL injection raises the alert')
        self.assertFalse(set(alert) & set(safe), 'a safe diff gets no alert')

    @unittest.skipUnless(os.getenv('QDRANT_URL'), 'Set QDRANT_URL')
    def test_hr_answers_from_qdrant(self):
        self.addCleanup(requests.delete, f"{os.environ['QDRANT_URL']}/collections/hr_policies", timeout=30)
        with open('hr_policy.md', 'w') as file:
            file.write(HR_POLICY)
        flow = self.build(self.vibe(), (
            'Answer employee questions about our HR policy. Keep the policy in our Qdrant server (its URL is in the '
            'QDRANT_URL environment variable, collection hr_policies), load ./hr_policy.md into it, and cite the '
            'sources.'))
        self.assert_assistants()
        stores = [self.options(t)['knowledge'] for t in self.language_tasks() if self.options(t).get('knowledge')]
        self.assertEqual((stores[0]['type'], stores[0]['url'], stores[0]['collection']),
                         ('qdrant', '${ENV:QDRANT_URL}', 'hr_policies'))

        self.assertIn('25', self.text(self.run_flow(flow, 'How many vacation days do I get?')))

    @unittest.skipUnless(os.getenv('GEMINI_API_KEY'), 'Set GEMINI_API_KEY')
    def test_web_facts(self):
        flow = self.build(self.vibe(), 'Answer questions about current events in two sentences, with web sources.')
        self.assert_assistants()
        self.assertTrue(any(self.params(t).get('google_search') for t in self.language_tasks()))

        output = self.run_flow(flow, 'What is the latest stable version of Python?')

        self.assertIn('sources', self.text(output))

    # ------------------------------------------------------------------ images and audio
    @unittest.skipUnless(os.getenv('VIBE_MEDIA_TESTS') == '1', 'Set VIBE_MEDIA_TESTS=1 (billed image generation)')
    def test_content_platform(self):
        flow = self.build(self.vibe(), (
            'I am building a blogging platform about the environment. Identify the requirements for the website, '
            'then write the website description and theme details, then a short image description for a logo and '
            'generate the logo image with OpenAI, and generate the website code (one HTML page) from the '
            'requirements and the description.'))
        self.assert_assistants()
        images = self.tasks('image')
        self.assertEqual(len(images), 1)

        output = self.run_flow(flow)

        self.assertGreater(len(output[images[0]['name']]['output']), 1000)
        self.assertIn('<html', self.text(output))

    @unittest.skipUnless(os.getenv('VIBE_MEDIA_TESTS') == '1' and os.getenv('MISTRAL_API_KEY'),
                         'Set VIBE_MEDIA_TESTS=1 and MISTRAL_API_KEY')
    def test_travel_assistant(self):
        flow = self.build(self.vibe(), (
            'A travel assistant: create a 3-day itinerary for the city in the input, read the first day aloud as an '
            'audio guide, make a picture of the destination and check which landmarks it shows, then write a '
            'complete travel guide that combines the itinerary and the picture analysis. Use Mistral for the final '
            'guide.'))
        self.assert_assistants()
        self.assertTrue({'speech', 'image', 'vision'} <= {t['agent']['agent_type'] for t in self.spec['tasks']})
        self.assertTrue(any(t['agent']['provider'] == 'mistral' for t in self.language_tasks()))

        output = self.run_flow(flow, 'Rome, Italy')

        audio = [item['output'] for item in output.values() if item['type'] == 'audio']
        self.assertGreater(len(audio[0]), 10000)
        vision = [output[t['name']]['output'] for t in self.tasks('vision')]
        self.assertGreater(len(str(vision[0])), 50)

    @unittest.skipUnless(os.getenv('VIBE_MEDIA_TESTS') == '1' and os.getenv('GEMINI_API_KEY'),
                         'Set VIBE_MEDIA_TESTS=1 and GEMINI_API_KEY')
    def test_news_digest(self):
        flow = self.build(self.vibe(), (
            'Every morning, find the three most important AI news stories of the last day on the web, write a short '
            'digest with sources, translate it to Arabic, and make an audio version of the Arabic digest.'))
        self.assert_assistants()
        self.assertTrue(any(self.params(t).get('google_search') for t in self.language_tasks()))

        output = self.run_flow(flow)

        audio = [item['output'] for item in output.values() if item['type'] == 'audio']
        self.assertGreater(len(audio[0]), 10000)
        self.assertTrue(any('؀' <= ch <= 'ۿ' for ch in self.text(output)))

    # ------------------------------------------------------------------ computer use on a local staging site
    @unittest.skipUnless(os.getenv('VIBE_COMPUTER_TESTS') == '1' and os.getenv('ANTHROPIC_API_KEY'),
                         'Set VIBE_COMPUTER_TESTS=1 and ANTHROPIC_API_KEY')
    def test_release_checks(self):
        server = site.start()
        self.addCleanup(server.shutdown)
        site.reset()
        GUARD_CALLS.clear()
        url = f'http://127.0.0.1:{server.server_port}'
        flow = self.build(self.vibe(guards={'read_only': read_only}), (
            f'Run smoke checks on our staging shop at {url} before each release. Journeys: check that the sign in '
            'page accepts the demo account shown on it; search for "laptop stand" and report how many results '
            'appear; add the first laptop stand to the cart and report the cart total and whether the checkout '
            'button is enabled. The checks must never place an order. Then write a short release check summary '
            'that clearly marks every failing journey.'))
        journeys = self.tasks('computer')
        self.assertGreaterEqual(len(journeys), 3)
        self.assertTrue(all(self.params(t).get('on_action') == 'read_only' for t in journeys))
        self.assertTrue(all(str(self.params(t).get('start_url', '')).startswith(url) for t in journeys))
        self.assert_assistants()

        text = self.text(self.run_flow(flow))

        self.assertTrue('3 results' in text or 'three results' in text or '3 laptop' in text)
        self.assertIn('49', text)
        self.assertTrue(site.STATE['signed_in'])
        self.assertEqual(site.STATE['orders'], 0)
        self.assertGreater(len(GUARD_CALLS), 3)

    @unittest.skipUnless(os.getenv('VIBE_COMPUTER_TESTS') == '1' and os.getenv('ANTHROPIC_API_KEY'),
                         'Set VIBE_COMPUTER_TESTS=1 and ANTHROPIC_API_KEY')
    def test_portal_operations(self):
        server = site.start()
        self.addCleanup(server.shutdown)
        site.reset()
        url = f'http://127.0.0.1:{server.server_port}/portal/invoices'
        flow = self.build(self.vibe(guards={'read_only': read_only}), (
            f'Every morning, open our supplier portal at {url} and read every invoice of this month with its '
            'number, date, amount and payment status. Then convert the list into JSON rows with the keys number, '
            'date, amount and status, so we can load it into our ERP. Only read the portal: never pay, submit or '
            'approve anything.'))
        self.assertTrue(all(self.params(t).get('on_action') == 'read_only' for t in self.tasks('computer')))
        self.assert_assistants()

        output = self.run_flow(flow)

        last = [t['name'] for t in self.spec['tasks'] if not (self.spec.get('map_paths') or {}).get(t['name'])]
        raw = ' '.join(str(output[name]['output']) for name in last if name in output)
        rows = json.loads(raw[raw.index('['):raw.rindex(']') + 1])
        self.assertEqual(len(rows), 5)
        self.assertLessEqual({'number', 'date', 'amount', 'status'}, set(rows[0]))
        self.assertEqual(site.STATE['payments'], 0)


if __name__ == '__main__':
    unittest.main()
