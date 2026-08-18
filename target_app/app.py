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
    if not member_id:
        return render_template(
            "search.html", error="Please enter a Member ID.", member_id=member_id
        )
    return redirect(url_for("member_detail", member_id=member_id))


@app.route("/advanced-search")
def advanced_search():
    return render_template(
        "message.html",
        title="Advanced Search",
        message="Advanced search is not available in this environment.",
    )


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


@app.route("/member/<member_id>/history")
def member_history(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    return render_template(
        "message.html",
        title="Transaction History",
        message="Transaction history is not available in this environment.",
    )


@app.route("/member/<member_id>/statement")
def member_statement(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    return render_template(
        "message.html",
        title="Print Statement",
        message="Statement printing is not available in this environment.",
    )


@app.route("/member/<member_id>/contact-info")
def member_contact_info(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    return render_template(
        "message.html",
        title="Update Contact Info",
        message="Contact info editing is not available in this environment.",
    )


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


@app.route("/member/<member_id>/withdraw", methods=["GET", "POST"])
def withdraw_new(member_id):
    member = get_member(member_id)

    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    if member.status == "restricted":
        return render_template(
            "message.html",
            title="Access Restricted",
            message="Cannot process a withdrawal: this member's account is restricted.",
        )

    if request.method == "GET":
        return render_template("withdraw_form.html", member=member)

    amount_raw = request.form.get("amount", "").strip()
    method = request.form.get("method", "transfer").strip()
    memo = request.form.get("memo", "").strip()

    error = None
    amount = None
    try:
        amount = float(amount_raw)
        if amount <= 0:
            error = "Amount must be a positive number."
        elif amount > member.balance:
            error = "Insufficient funds for this withdrawal."
    except ValueError:
        error = "Amount must be a positive number."

    if error:
        return render_template(
            "withdraw_form.html",
            member=member,
            error=error,
            amount=amount_raw,
            method=method,
            memo=memo,
        )

    return render_template(
        "withdraw_confirm.html", member=member, amount=amount, method=method, memo=memo
    )


@app.route("/member/<member_id>/withdraw/confirm", methods=["POST"])
def withdraw_confirm(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    amount = float(request.form.get("amount", "0"))
    method = request.form.get("method", "transfer").strip()
    memo = request.form.get("memo", "").strip()

    # Deliberately no amount-cap check here
    member.balance -= amount

    return render_template(
        "withdraw_success.html",
        member_id=member_id,
        amount=amount,
        balance=member.balance,
        method=method,
        memo=memo,
    )


@app.route("/member/<member_id>/contact/confirm", methods=["POST"])
def contact_confirm(member_id):
    # Unrelated widget sharing the withdraw-confirm page, deliberately
    # using the same "Confirm" button role+name as the real withdrawal
    # action. Exists to force real locator disambiguation rather than a
    # trivially unique label - it must never be able to advance or affect
    # the withdrawal flow.
    member = get_member(member_id)
    if member is None:
        return render_template("member_not_found.html", member_id=member_id)

    return render_template(
        "message.html",
        title="Contact Verified",
        message="Contact information confirmed on file.",
    )


@app.route("/admin/delete-member/<member_id>", methods=["GET"])
def admin_delete_member_confirm(member_id):
    return render_template("admin_delete_confirm.html", member_id=member_id)


@app.route("/admin/delete-member/<member_id>", methods=["POST"])
def admin_delete_member(member_id):
    # Deliberately does not mutate MEMBERS.
    return render_template("admin_deleted.html", member_id=member_id)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
