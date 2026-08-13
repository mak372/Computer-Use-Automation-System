"""Flask app entry point for the simulated legacy bank back-office tool.

This app has no awareness of automation, guardrails, or risk policy - it
simply serves pages the way a real (unprotected) legacy internal tool
would. All safety, allowlist, and risk-tiering logic lives in the
orchestration layer that drives a browser against this app, not here.
"""

import time

from flask import Flask, redirect, render_template, request, url_for

from data import MAX_SUB_ACCOUNTS, SubAccount, get_member

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


@app.route("/member/<member_id>/sub-account/new", methods=["GET", "POST"])
def sub_account_new(member_id):
    member = get_member(member_id)

    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    if member.status == "restricted":
        return render_template(
            "message.html",
            title="Access Restricted",
            message="Cannot open a sub-account: this member's account is restricted.",
        )

    if len(member.sub_accounts) >= MAX_SUB_ACCOUNTS:
        return render_template(
            "message.html",
            title="Limit Reached",
            message=f"This member already has the maximum number of sub-accounts ({MAX_SUB_ACCOUNTS}).",
        )

    if request.method == "GET":
        return render_template("sub_account_form.html", member_id=member_id)

    account_type = request.form.get("account_type", "savings")
    nickname = request.form.get("nickname", "").strip()
    deposit_amount_raw = request.form.get("deposit_amount", "").strip()
    interstitial_ack = request.form.get("interstitial_ack") == "1"

    error = None
    try:
        deposit_amount = float(deposit_amount_raw)
        if deposit_amount <= 0:
            error = "Initial deposit must be a positive number."
    except ValueError:
        error = "Initial deposit must be a positive number."

    if not nickname:
        error = "Nickname is required."

    if error:
        return render_template(
            "sub_account_form.html",
            member_id=member_id,
            error=error,
            account_type=account_type,
            nickname=nickname,
            deposit_amount=deposit_amount_raw,
        )

    if member.similar_account_exists and not interstitial_ack:
        return render_template(
            "sub_account_interstitial.html",
            member_id=member_id,
            account_type=account_type,
            nickname=nickname,
            deposit_amount=deposit_amount_raw,
        )

    return render_template(
        "sub_account_confirm.html",
        member_id=member_id,
        account_type=account_type,
        nickname=nickname,
        deposit_amount=deposit_amount_raw,
    )


@app.route("/member/<member_id>/sub-account/confirm", methods=["POST"])
def sub_account_confirm(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    account_type = request.form.get("account_type", "savings")
    nickname = request.form.get("nickname", "").strip()
    deposit_amount = float(request.form.get("deposit_amount", "0"))

    member.sub_accounts.append(
        SubAccount(name=nickname, account_type=account_type, balance=deposit_amount)
    )

    return render_template(
        "sub_account_success.html",
        member_id=member_id,
        account_type=account_type,
        nickname=nickname,
        deposit_amount=deposit_amount,
    )


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
