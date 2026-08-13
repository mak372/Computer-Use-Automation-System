"""Flask app entry point for the simulated legacy bank back-office tool.

This app has no awareness of automation, guardrails, or risk policy - it
simply serves pages the way a real (unprotected) legacy internal tool
would. All safety, allowlist, and risk-tiering logic lives in the
orchestration layer that drives a browser against this app, not here.
"""

import time

from flask import Flask, redirect, render_template, request, url_for

from data import get_member

app = Flask(__name__)

SLOW_DELAY_SECONDS = 3


@app.route("/")
def search():
    return render_template("search.html")


@app.route("/lookup")
def lookup():
    member_id = request.args.get("member_id", "").strip()
    return redirect(url_for("member_detail", member_id=member_id))


@app.route("/member/<member_id>")
def member_detail(member_id):
    member = get_member(member_id)

    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    if member.broken:
        return render_template("member_broken.html", member_id=member_id)

    if member.force_timeout:
        return render_template("session_expired.html", member_id=member_id)

    if member.status == "restricted":
        return render_template("member_restricted.html", member=member)

    if member.slow:
        time.sleep(SLOW_DELAY_SECONDS)

    return render_template("member_detail.html", member=member)


@app.route("/admin/delete-member/<member_id>", methods=["GET"])
def admin_delete_member_confirm(member_id):
    return render_template("admin_delete_confirm.html", member_id=member_id)


@app.route("/admin/delete-member/<member_id>", methods=["POST"])
def admin_delete_member(member_id):
    # Deliberately does not mutate MEMBERS. This route exists only as an
    # off-allowlist decoy to demonstrate guardrail enforcement - it must
    # never actually be reached by the automated agent.
    return render_template("admin_deleted.html", member_id=member_id)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
